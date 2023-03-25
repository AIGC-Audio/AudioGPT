import chardet


def get_encoding(file):
    with open(file, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    if encoding == 'GB2312':
        encoding = 'GB18030'
    return encoding
