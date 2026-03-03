import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap
import dtlpy as dl
import logging
import json
import re
import os
import tempfile
import threading
from datetime import datetime

logger = logging.getLogger("[GRAPH-RAG]")

SAVE_INTERVAL_SEC = 5 * 60
GRAPH_PATH = "/graph_rag"
GRAPH_FILENAME = "knowledge_graph.json"
# ====================================================================== #
#  add_chunk_to_graph accepts two input formats:                         #
#                                                                        #
#  1. Prompt item — LLM guided-JSON response.                            #
#     assistant msg = JSON matching GRAPH_EXTRACTION_SCHEMA              #
#                                                                        #
#  2. JSON file item:                                                    #
#     {"chunk_name", "text", "entities": [...], "relationships": [...]}  #
#                                                                        #
#  One graph is maintained per dataset (knowledge_graph.json).           #
# ====================================================================== #

GRAPH_EXTRACTION_PROMPT = (
    "You are a knowledge-graph extraction engine. "
    "Given a text passage, extract the most important entities and "
    "the relationships between them.\n\n"
    "Entity rules:\n"
    "- Use Title Case canonical names (\"Assembly Line\", not \"assembly-line\").\n"
    "- Merge synonyms into one canonical name "
    "(pick the most common form, e.g. \"Car\" not \"car/vehicle/automobile\").\n"
    "- SKIP low-value entities: URLs, watermarks, colors, directions, "
    "generic sizes, timestamps.\n"
    "- Focus on meaningful nouns: people, objects, places, organizations, "
    "concepts, events.\n"
    "- Prefer specific names over generic ones "
    "(\"Forklift\" not \"Vehicle\", \"Warehouse\" not \"Building\").\n\n"
    "Relationship rules:\n"
    "- \"source\" and \"target\" MUST exactly match an entity \"name\".\n"
    "- \"relation\" must be a short UPPER_SNAKE_CASE verb "
    "(e.g. LOCATED_IN, OPERATES, CONTAINS, PART_OF, CAUSES, PRODUCES).\n"
    "- Only include relationships clearly stated or strongly implied.\n"
    "- Do NOT invent relationships that require speculation.\n\n"
    "Return valid JSON matching the provided schema."
)

GRAPH_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Canonical name of the entity",
                    },
                    "type": {
                        "type": "string",
                        "description": "Entity type, e.g. Person, Object, Location, "
                        "Organisation, Concept, Event, Equipment, Attribute",
                    },
                },
                "required": ["name", "type"],
            },
            "minItems": 2,
            "maxItems": 8,
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Must match an entity name exactly",
                    },
                    "target": {
                        "type": "string",
                        "description": "Must match an entity name exactly",
                    },
                    "relation": {
                        "type": "string",
                        "description": "UPPER_SNAKE_CASE verb, e.g. PLACES, CAUSES, LOCATED_IN",
                    },
                    "description": {
                        "type": "string",
                        "description": "Free-text description of this relationship",
                    },
                },
                "required": ["source", "target", "relation"],
            },
            "minItems": 1,
            "maxItems": 10,
        },
    },
    "required": ["entities", "relationships"],
}


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "about", "above",
    "after", "again", "all", "also", "and", "any", "because", "before",
    "between", "both", "but", "by", "each", "for", "from", "get", "got",
    "how", "if", "in", "into", "it", "its", "just", "like", "more",
    "most", "not", "now", "of", "on", "only", "or", "other", "our",
    "out", "over", "own", "same", "she", "so", "some", "such", "than",
    "that", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very",
    "what", "when", "where", "which", "while", "who", "whom", "why",
    "with", "you", "your", "here", "just", "much", "many", "no", "nor",
    "yes", "yet", "tell", "show", "find", "give", "describe", "explain",
    "happening", "happened", "going", "does", "doing",
}


class ServiceRunner(dl.BaseServiceRunner):

    def __init__(self):
        super().__init__()
        self._graphs: dict[str, nx.DiGraph] = {}
        self._dirty: dict[str, bool] = {}
        self._datasets: dict[str, dl.Dataset] = {}
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._saver_thread = threading.Thread(
            target=self._background_saver, daemon=True,
        )
        self._saver_thread.start()
        logger.info("Graph-RAG service initialised, background saver started")

    # ------------------------------------------------------------------ #
    #  Per-dataset graph cache                                             #
    # ------------------------------------------------------------------ #
    def _get_graph(self, dataset: dl.Dataset) -> nx.DiGraph:
        """Return the in-memory graph for *dataset*, loading on first access."""
        ds_id = dataset.id
        if ds_id not in self._graphs:
            with self._lock:
                if ds_id not in self._graphs:
                    self._graphs[ds_id] = self._download_graph(dataset)
                    self._dirty[ds_id] = False
                    self._datasets[ds_id] = dataset
        return self._graphs[ds_id]

    def _mark_dirty(self, dataset_id: str):
        self._dirty[dataset_id] = True

    # ------------------------------------------------------------------ #
    #  Background saver — uploads every SAVE_INTERVAL_SEC if dirty         #
    # ------------------------------------------------------------------ #
    def _background_saver(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=SAVE_INTERVAL_SEC)
            self._flush_dirty_graphs()

    def _flush_dirty_graphs(self):
        for ds_id in list(self._dirty):
            if not self._dirty.get(ds_id):
                continue
            with self._lock:
                # re-check after acquiring lock — another thread may have saved it
                if not self._dirty.get(ds_id):
                    continue
                G = self._graphs[ds_id]
                dataset = self._datasets[ds_id]
                self._dirty[ds_id] = False
            try:
                self._upload_graph(G, dataset)
                self._visualize_and_upload(G, dataset)
                logger.info(
                    f"Background save: dataset {ds_id} — "
                    f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
                )
            except Exception:
                logger.exception(f"Background save failed for dataset {ds_id}")
                self._dirty[ds_id] = True

    # ------------------------------------------------------------------ #
    #  Graph download / upload helpers                                     #
    # ------------------------------------------------------------------ #
    def _download_graph(self, dataset: dl.Dataset) -> nx.DiGraph:
        try:
            filters = dl.Filters()
            filters.add(field="name", values=self.GRAPH_FILENAME)
            filters.add(field="dir", values=GRAPH_PATH)
            pages = dataset.items.list(filters=filters)
            for graph_item in pages.all():
                buf = graph_item.download(save_locally=False)
                data = json.loads(buf.read().decode("utf-8"))
                G = nx.node_link_graph(data)
                logger.info(
                    f"Loaded graph for dataset {dataset.id}: "
                    f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
                )
                return G
        except Exception as e:
            logger.info(f"No existing graph for dataset {dataset.id} ({e}), creating new")
        return nx.DiGraph()

    def _upload_graph(self, G: nx.DiGraph, dataset: dl.Dataset) -> dl.Item:
        data = nx.node_link_data(G)
        data["_meta"] = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        )
        try:
            json.dump(data, tmp, indent=2)
            tmp.close()
            return dataset.items.upload(
                local_path=tmp.name,
                remote_name=self.GRAPH_FILENAME,
                remote_path=GRAPH_PATH,
                overwrite=True,
                item_metadata={
                    "user": {
                        "type": "knowledge_graph",
                        "num_nodes": G.number_of_nodes(),
                        "num_edges": G.number_of_edges(),
                    }
                },
            )
        finally:
            os.remove(tmp.name)

    # ------------------------------------------------------------------ #
    #  Merge structured data into the graph                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize entity name: strip, collapse whitespace, title case."""
        name = re.sub(r"[-_]+", " ", name.strip())
        name = re.sub(r"\s+", " ", name)
        return name.title()

    @staticmethod
    def _merge_into_graph(
        G: nx.DiGraph,
        chunk_name: str,
        text: str,
        item_id: str,
        entities: list[dict],
        relationships: list[dict],
    ):
        chunk_node = f"Chunk:{chunk_name}"
        G.add_node(chunk_node, type="chunk", text=text, item_id=item_id)

        entity_map: dict[str, str] = {}
        for ent in entities:
            raw_name = ent.get("name", "").strip()
            etype = ent.get("type", "Entity").strip()
            if not raw_name:
                continue
            name = ServiceRunner._normalize_name(raw_name)
            nid = f"{etype.title()}:{name}"
            if nid not in G:
                G.add_node(nid, type=etype.lower(), label=name)
            G.add_edge(chunk_node, nid, label="MENTIONS")
            entity_map[raw_name.lower()] = nid
            entity_map[name.lower()] = nid

        for rel in relationships:
            src = rel.get("source", "").strip()
            tgt = rel.get("target", "").strip()
            relation = rel.get("relation", "RELATED_TO").strip().upper()
            desc = rel.get("description", "")
            src_id = entity_map.get(src.lower()) or entity_map.get(
                ServiceRunner._normalize_name(src).lower()
            )
            tgt_id = entity_map.get(tgt.lower()) or entity_map.get(
                ServiceRunner._normalize_name(tgt).lower()
            )
            if src_id and tgt_id and src_id != tgt_id:
                G.add_edge(src_id, tgt_id, label=relation, description=desc)

    # ------------------------------------------------------------------ #
    #  1. Build graph — incremental, one item at a time                   #
    # ------------------------------------------------------------------ #
    def add_chunk_to_graph(self, item: dl.Item) -> dl.Item:
        """
        Pipeline node — accepts one of:

        • **Prompt item** with an LLM response (guided JSON) as the last
          assistant message containing {entities[], relationships[]}.
          The user message is used as the chunk text.

        • **JSON item** (.json) with the structured schema:
          {chunk_name, text, entities[], relationships[]}

        Raises ValueError for unsupported item formats.
        A single graph is maintained per dataset.
        """
        chunk_name, text, entities, relationships = self._parse_item(item)

        dataset = item.dataset
        G = self._get_graph(dataset)

        with self._lock:
            self._merge_into_graph(G, chunk_name, text, item.id, entities, relationships)
            self._mark_dirty(dataset.id)

        logger.info(
            f"Added chunk {chunk_name!r} to graph (dataset {dataset.id}) "
            f"— {len(entities)} entities, {len(relationships)} relations"
        )
        return item

    @staticmethod
    def _is_prompt_item(item: dl.Item) -> bool:
        return (
            item.metadata.get("system", {})
            .get("shebang", {})
            .get("dltype")
            == "prompt"
        )  # TODO: IF JSON - WHETER A PRPMOT, NO NEEED FOR THIS FUNCTION

    @staticmethod
    def _parse_item(item: dl.Item) -> tuple[str, str, list[dict], list[dict]]:
        """
        Extract (chunk_name, text, entities, relationships) from an item.
        Supports prompt items and structured JSON items only.
        Raises ValueError for any other format.
        """
        if ServiceRunner._is_prompt_item(item):
            return ServiceRunner._parse_prompt_item(item)

        mimetype = item.metadata.get("system", {}).get("mimetype", "")
        if mimetype.startswith("application/json") or item.name.endswith(".json"): #TODO: change this one
            return ServiceRunner._parse_json_item(item)

        raise ValueError(
            f"Unsupported item format for '{item.name}' (mimetype={mimetype}). "
            f"Expected a prompt item or a .json file."
        )

    @staticmethod
    def _parse_prompt_item(item: dl.Item) -> tuple[str, str, list[dict], list[dict]]:
        """Parse a prompt item — user message = text, assistant message = guided JSON."""
        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages()

        user_text = ""
        assistant_raw = None
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", [])
            if not content:
                continue
            value = content[0].get("text", "")
            if role == "user" and value:
                user_text = value
            elif role == "assistant" and value:
                assistant_raw = value

        if not assistant_raw:
            raise ValueError(
                f"Prompt item '{item.name}' has no assistant response to extract."
            )
        # TODO: EITHER 

        data = ServiceRunner._extract_json(assistant_raw)
        entities, relationships = ServiceRunner._split_entities_and_relationships(data)
        return (
            item.name,
            user_text,
            entities,
            relationships,
        ) # todo: do inned this?

    @staticmethod
    def _extract_json(text: str): #todo: check if there is a function for it
        """Extract JSON from a raw LLM response, stripping markdown fences and surrounding text."""
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        for start in range(len(text)):
            if text[start] in ("{", "["):
                bracket = "}" if text[start] == "{" else "]"
                for end in range(len(text) - 1, start - 1, -1):
                    if text[end] == bracket:
                        return json.loads(text[start:end + 1])

        raise ValueError("No valid JSON found in LLM response.")

    @staticmethod
    def _split_entities_and_relationships(data) -> tuple[list[dict], list[dict]]:
        """
        Handle both structured {entities, relationships} and flat-array
        formats where entities and relationships are mixed in one list.
        """
        if isinstance(data, dict):
            return data.get("entities", []), data.get("relationships", [])

        if isinstance(data, list):
            entities = []
            relationships = []
            for obj in data:
                if "source" in obj and "target" in obj:
                    relationships.append(obj)
                elif "name" in obj:
                    entities.append(obj)
            return entities, relationships

        raise ValueError(f"Unexpected JSON type: {type(data).__name__}")

    @staticmethod
    def _parse_json_item(item: dl.Item) -> tuple[str, str, list[dict], list[dict]]: 
        """Parse a structured JSON item with entities and relationships."""
        buf = item.download(save_locally=False)
        raw = buf.read().decode("utf-8", errors="replace").strip()
        if not raw:
            raise ValueError(f"JSON item '{item.name}' is empty.")

        data = json.loads(raw)
        return (
            data.get("chunk_name", item.name),
            data.get("text", ""),
            data.get("entities", []),
            data.get("relationships", []),
        )
        # TODO: 1 RETURN

    # ------------------------------------------------------------------ #
    #  2. Retrieve from graph — structured + keyword query                 #
    # ------------------------------------------------------------------ #
    def query_graph(
        self,
        item: dl.Item,
        dataset: dl.Dataset,
        entity_name: str = None,
        relationship: str = None,
        target_name: str = None,
        hops: int = 2,
    ) -> dl.Item:
        """
        Pipeline node — receives a prompt item, searches the dataset
        knowledge graph, and adds retrieved context to the prompt.

        Supports two modes:

        **Structured** (Cypher-like) — when any of ``entity_name``,
        ``relationship``, or ``target_name`` are provided, edges are
        filtered precisely, equivalent to::

            MATCH (source)-[r:RELATIONSHIP]->(target)
            WHERE source.label =~ entity_name
              AND target.label =~ target_name

        ``*`` wildcards are supported (e.g. ``warehouse*``).

        **Keyword** (default) — extracts keywords from the user message
        and matches both entity labels *and* relationship types.

        In both modes the matched sub-graph is expanded via BFS up to
        ``hops`` levels to collect source chunk texts.
        """
        query_text = self._extract_query_from_prompt(item)

        G = self._load_graph(dataset)
        if G.number_of_nodes() == 0:
            logger.warning("No graph data available in this dataset.")
            return item

        if entity_name or relationship or target_name:
            matched_edges, chunks = self._structured_query(
                G, entity_name, relationship, target_name, hops,
            )
            logger.info(
                f"Structured query (entity={entity_name}, rel={relationship}, "
                f"target={target_name}) -> {len(matched_edges)} edges, "
                f"{len(chunks)} chunks"
            )
        else:
            if not query_text:
                logger.warning(f"No user message in prompt item {item.id}")
                return item 
            matched_edges, chunks = self._keyword_query(
                G, query_text, hops,
            )

        if not matched_edges and not chunks:
            return item

        context = self._build_context(
            G, query_text or "structured query", matched_edges, chunks,
        )

        source_items = [
            {"item_id": c["item_id"], "name": c["name"]}
            for c in chunks if c.get("item_id")
        ]

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
        )
        try:
            tmp.write(context)
            tmp.close()
            context_item = dataset.items.upload(
                local_path=tmp.name,
                remote_name=f"context-{item.name}--{datetime.now().strftime('%Y%m%d%H%M%S')}.txt",
                remote_path=item.dir,
                overwrite=True,
                item_metadata={
                    "user": {
                        "type": "graph_rag_context",
                        "source_query": query_text or "",
                        "num_triples": len(matched_edges),
                        "num_source_chunks": len(chunks),
                        "source_chunks": source_items,
                    }
                },
            )
        finally:
            os.remove(tmp.name)

        prompt_item = dl.PromptItem.from_item(item)
        prompt_item.prompts[-1].add_element(
            mimetype=dl.PromptType.METADATA,
            value={"nearestItems": [context_item.id]},
        )
        prompt_item.update()
        return item

    # ------------------------------------------------------------------ #
    #  Structured query — Cypher-like filtering                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _structured_query(
        G: nx.DiGraph,
        entity_name: str = None,
        relationship: str = None,
        target_name: str = None,
        hops: int = 1,
    ) -> tuple[list[tuple], list[dict]]:
        """
        Filter edges by source label, relationship type, and target label.

        Wildcard ``*`` in entity/target names is converted to ``.*`` for
        regex matching, so ``warehouse*`` matches ``Warehouse A``, etc.
        """
        def _matches(pattern: str | None, label: str) -> bool:
            if pattern is None:
                return True
            regex = re.escape(pattern).replace(r"\*", ".*")
            return bool(re.search(regex, label, re.IGNORECASE))

        matched_edges: list[tuple] = []
        seed_nodes: set[str] = set()

        for u, v, d in G.edges(data=True):
            edge_label = d.get("label", "")
            if edge_label == "MENTIONS":
                continue
            if relationship and edge_label.upper() != relationship.strip().upper():
                continue

            u_label = G.nodes[u].get("label", "") if u in G.nodes else ""
            v_label = G.nodes[v].get("label", "") if v in G.nodes else ""

            if not _matches(entity_name, u_label):
                continue
            if not _matches(target_name, v_label):
                continue

            matched_edges.append((u, v, d))
            seed_nodes.update({u, v})

        chunks = ServiceRunner._collect_chunks_bfs(G, seed_nodes, hops)
        return matched_edges, chunks

    # ------------------------------------------------------------------ #
    #  Keyword query — NL fallback with relationship-type awareness        #
    # ------------------------------------------------------------------ #
    def _keyword_query(
        self,
        G: nx.DiGraph,
        query_text: str,
        hops: int = 2,
    ) -> tuple[list[tuple], list[dict]]:
        """
        Extract keywords from the user query and match against both
        entity labels and relationship types in the graph.
        """
        keywords = self._extract_keywords(query_text)
        if not keywords:
            logger.info(f"No usable keywords in query: {query_text}")
            return [], []

        matched_nodes: set[str] = set()
        for nid, d in G.nodes(data=True):
            if d.get("type") == "chunk":
                continue
            label = d.get("label", "").lower()
            if any(kw in label for kw in keywords):
                matched_nodes.add(nid)

        matched_rels: set[str] = set()
        for _, _, d in G.edges(data=True):
            rel = d.get("label", "")
            if rel == "MENTIONS":
                continue
            rel_words = {w for w in rel.lower().split("_") if len(w) > 2}
            if rel_words & keywords:
                matched_rels.add(rel)

        logger.info(
            f"Keywords: {keywords} -> {len(matched_nodes)} entities, "
            f"{len(matched_rels)} relationship types ({matched_rels or 'all'})"
        )

        matched_edges: list[tuple] = []
        seed_nodes: set[str] = set()
        for u, v, d in G.edges(data=True):
            rel = d.get("label", "")
            if rel == "MENTIONS":
                continue
            if u not in matched_nodes and v not in matched_nodes:
                continue
            if matched_rels and rel not in matched_rels:
                continue
            matched_edges.append((u, v, d))
            seed_nodes.update({u, v})

        chunks = self._collect_chunks_bfs(G, seed_nodes, hops)
        return matched_edges, chunks

    # ------------------------------------------------------------------ #
    #  BFS chunk collector                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _collect_chunks_bfs( # TODO : WHETER THERE IS A FUNCTION TO BFS
        G: nx.DiGraph, seed_nodes: set[str], max_hops: int,
    ) -> list[dict]:
        """
        BFS from *seed_nodes* up to *max_hops*.

        Returns a list of dicts, one per discovered chunk node::

            {"node_id": "Chunk:frame_01.json", "item_id": "abc123",
             "name": "frame_01.json", "text": "..."}
        """
        visited = set(seed_nodes)
        frontier = set(seed_nodes)
        chunks: list[dict] = []
        seen_chunks: set[str] = set()

        def _try_add(node_id: str):
            if node_id in seen_chunks:
                return
            nd = G.nodes.get(node_id, {})
            if nd.get("type") != "chunk":
                return
            seen_chunks.add(node_id)
            chunks.append({
                "node_id": node_id,
                "item_id": nd.get("item_id", ""),
                "name": node_id.split(":", 1)[-1] if ":" in node_id else node_id,
                "text": nd.get("text", ""),
            })

        for nd in seed_nodes:
            _try_add(nd)

        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                neighbors = set(G.predecessors(node)) | set(G.successors(node))
                for nb in neighbors:
                    if nb in visited:
                        continue
                    visited.add(nb)
                    next_frontier.add(nb)
                    _try_add(nb)
            frontier = next_frontier

        return chunks

    # ------------------------------------------------------------------ #
    #  Context formatting                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_context(
        G: nx.DiGraph,
        query: str,
        edges: list[tuple],
        chunks: list[dict],
    ) -> str:
        """
        Format graph retrieval results as an LLM-readable context block.

        Output includes:
        - Typed entity-relationship triples grouped by source entity
        - Source passages with provenance (chunk name / item ID)
        """
        lines = [
            "=== Graph-RAG Context ===",
            f"User query: {query}",
        ]

        # -- Typed triples grouped by source entity --
        by_source: dict[str, list[str]] = {}
        seen: set[tuple] = set()
        for u, v, d in edges:
            u_data = G.nodes.get(u, {})
            v_data = G.nodes.get(v, {})
            u_type = u_data.get("type", "entity").capitalize()
            v_type = v_data.get("type", "entity").capitalize()
            u_lbl = u_data.get("label", u)
            v_lbl = v_data.get("label", v)
            rel = d.get("label", "RELATED")
            desc = d.get("description", "")

            key = (u_lbl, rel, v_lbl)
            if key in seen:
                continue
            seen.add(key)

            triple = f"({u_lbl}:{u_type})-[:{rel}]->({v_lbl}:{v_type})"
            if desc:
                triple += f"  // {desc}"
            by_source.setdefault(u_lbl, []).append(triple)

        if by_source:
            lines += ["", "Structured facts:"]
            for source, triples in by_source.items():
                lines.append(f"  [{source}]")
                for t in triples:
                    lines.append(f"    {t}")

        # -- Source passages with provenance --
        if chunks:
            lines += ["", "Source passages:"]
            for i, chunk in enumerate(chunks[:10], 1):
                name = chunk.get("name", "unknown")
                item_id = chunk.get("item_id", "")
                text = chunk.get("text", "")[:500]
                ref = f"source={name}"
                if item_id:
                    ref += f", item_id={item_id}"
                lines.append(f"  [{i}] ({ref})")
                if text:
                    lines.append(f"      {text}")

        lines.append("=== End Context ===")
        return "\n".join(lines)

    @staticmethod
    def _extract_query_from_prompt(item: dl.Item) -> str:
        """Extract the last user message text from a Dataloop PromptItem."""
        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages(include_assistant=False)
        if not messages:
            return ""
        last_message = messages[-1]
        content = last_message.get("content", [])
        if not content:
            return ""
        return content[0].get("text", "")

    @staticmethod
    def _extract_keywords(query: str) -> set[str]:
        """Extract meaningful keywords from a query, filtering stop words."""
        words = re.findall(r"[a-zA-Z0-9]+", query.lower())
        return {w for w in words if len(w) > 2 and w not in STOP_WORDS}

    # ------------------------------------------------------------------ #
    #  Visualize & upload (called automatically on every background save)  #
    # ------------------------------------------------------------------ #
    def _visualize_and_upload(
        self, G: nx.DiGraph, dataset: dl.Dataset,
    ) -> dl.Item:
        TYPE_STYLES = {
            "chunk":        {"color": "#90CAF9", "size": 1400, "shape": "o", "edge": "#1565C0"},
            "person":       {"color": "#EF9A9A", "size": 1100, "shape": "o", "edge": "#C62828"},
            "object":       {"color": "#81C784", "size": 1000, "shape": "s", "edge": "#2E7D32"},
            "equipment":    {"color": "#FFD54F", "size": 1000, "shape": "h", "edge": "#F57F17"},
            "location":     {"color": "#CE93D8", "size": 1000, "shape": "d", "edge": "#6A1B9A"},
            "event":        {"color": "#FFAB91", "size": 1000, "shape": "^", "edge": "#BF360C"},
            "attribute":    {"color": "#B0BEC5", "size": 800,  "shape": "o", "edge": "#455A64"},
            "organisation": {"color": "#80DEEA", "size": 1100, "shape": "s", "edge": "#00838F"},
            "concept":      {"color": "#FFF59D", "size": 900,  "shape": "d", "edge": "#F9A825"},
        }
        DEFAULT_STYLE = {"color": "#E0E0E0", "size": 900, "shape": "o", "edge": "#616161"}

        fig, ax = plt.subplots(figsize=(26, 18))

        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "Empty graph", ha="center", va="center",
                    fontsize=18, color="gray")
        else:
            k = 3.0 / (G.number_of_nodes() ** 0.5) if G.number_of_nodes() > 1 else 1.0
            pos = nx.spring_layout(G, seed=42, k=k, iterations=100)

            drawn_types = set()
            for nid, d in G.nodes(data=True):
                ntype = d.get("type", "").lower()
                drawn_types.add(ntype)
                style = TYPE_STYLES.get(ntype, DEFAULT_STYLE)
                nx.draw_networkx_nodes(
                    G, pos, nodelist=[nid], ax=ax,
                    node_color=style["color"], node_size=style["size"],
                    node_shape=style["shape"], alpha=0.92,
                    edgecolors=style["edge"], linewidths=1.2,
                )

            labels = {}
            for n, d in G.nodes(data=True):
                lbl = d.get("label", n.split(":")[-1] if ":" in n else n)
                labels[n] = "\n".join(textwrap.wrap(str(lbl), width=14))
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                                    font_size=6, font_weight="bold")

            mention_edges = [(u, v) for u, v, d in G.edges(data=True)
                             if d.get("label") == "MENTIONS"]
            relation_edges = [(u, v) for u, v, d in G.edges(data=True)
                              if d.get("label") != "MENTIONS"]

            if mention_edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=mention_edges, ax=ax,
                    arrowstyle="-|>", arrowsize=8,
                    edge_color="#90CAF9", alpha=0.3, style="dashed",
                )
            if relation_edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=relation_edges, ax=ax,
                    arrowstyle="-|>", arrowsize=12,
                    edge_color="#455A64", alpha=0.7,
                    connectionstyle="arc3,rad=0.1",
                )

            edge_labels = {
                (u, v): d.get("label", "")
                for u, v, d in G.edges(data=True)
                if d.get("label") != "MENTIONS"
            }
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                         font_color="#1565C0", font_size=5)

            legend_items = []
            for tname, style in TYPE_STYLES.items():
                if tname in drawn_types:
                    marker = {"o": "o", "s": "s", "h": "h", "d": "D", "^": "^"}.get(
                        style["shape"], "o"
                    )
                    legend_items.append(
                        plt.Line2D([], [], marker=marker, color="w",
                                   markerfacecolor=style["color"], markersize=10,
                                   label=tname.capitalize())
                    )
            if legend_items:
                ax.legend(handles=legend_items, loc="upper left",
                          fontsize=9, framealpha=0.9)

        title = f"Knowledge Graph - {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
        ax.axis("off")
        fig.tight_layout()

        img_name = "knowledge_graph.png"
        local_path = os.path.join(tempfile.gettempdir(), img_name)
        fig.savefig(local_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        try:
            uploaded = dataset.items.upload(
                local_path=local_path,
                remote_name=img_name,
                remote_path=GRAPH_PATH,
                overwrite=True,
                item_metadata={
                    "user": {
                        "type": "knowledge_graph_visualization",
                        "num_nodes": G.number_of_nodes(),
                        "num_edges": G.number_of_edges(),
                    }
                },
            )
        finally:
            os.remove(local_path)
        return uploaded

