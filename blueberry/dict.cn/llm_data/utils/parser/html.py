

def space_like_removed(text):
    text = text.strip().replace(' ', '').replace('\n', '').replace('&nbsp;', '').replace('\xa0', '')
    return text