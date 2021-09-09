def get_lines():
    return len(open('recognizer/tools/dictionary/chars.txt', 'r',encoding='utf-8').readlines())


num_class = get_lines() + 1


