import os,sys,shutil

def enclosed_by_dash_lines(text:str, factor=1) -> str:
    '''
    To make easy to see log file, emphasize the sentence by lines
    '''
    return f'{"-"*len(text)*factor}\n{text}\n{"-"*len(text)*factor}'

def create_or_clear_directory(directory_path) -> None:
    '''
    Make a directory as a vacant folder
    '''
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)
