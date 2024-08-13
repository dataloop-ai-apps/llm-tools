import json

import dtlpy as dl


def clean():
    dpk = dl.dpks.get(dpk_name='llm-tools-retriever-58171757-242c-4b08-b696-0e504d6bf0f5')
    filters = dl.Filters(field='dpkName', values=dpk.name, resource='apps')

    for app in dl.apps.list(filters=filters).all():
        print(app.name, app.project.name)
        app.uninstall()

    [i.delete() for i in list(dpk.revisions.all())]


def publish_and_install(project):
    # publish dpk to app store
    dpk = project.dpks.publish(manifest_filepath='retriever/dataloop.json', ignore_max_file_size=True)
    print(f'published successfully! dpk name: {dpk.name}, version: {dpk.version}, dpk id: {dpk.id}')
    try:
        app = project.apps.get(app_name=dpk.display_name)
        print(f'already installed, updating...')
        app.dpk_version = dpk.version
        app.update()
        print(f'update done. app id: {app.id}')
    except dl.exceptions.NotFound:
        print(f'installing ..')

        app = project.apps.install(dpk=dpk,
                                   app_name=dpk.display_name,
                                   scope=dl.AppScope.PROJECT)
        print(f'installed! app id: {app.id}')
    print(f'Done!')


if __name__ == "__main__":
    dl.setenv('prod')
    project = dl.projects.get(project_name="Merkava Demo")
    # project = dl.projects.get(project_id="8e3a66ea-f99d-40c5-9090-5f491df559aa")
    publish_and_install(project=project)
