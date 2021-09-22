

if __name__ == '__main__':

    src_train_file_path = 'tmp_data/recognizer_txts/train.txt'
    dst_train_file_path = 'tmp_data/recognizer_txts/real_train.txt'
    dictionary_file_path = config.char_path
    char_to_index = dict()

# Reading Chars as Dictionary

    with open(dictionary_file_path, 'r', encoding='utf-8') as in_file:
        lines = in_file.readlines()
        for index, line in enumerate(lines):
            line = line.strip('\r').strip('\n')
            char_to_index[line] = index


## Train


    with open(dst_train_file_path, 'a') as out_file:
            with open(src_train_file_path, 'r', encoding='utf-8') as in_file:
                lines = in_file.readlines()
                for index, line in enumerate(lines):
                    line = line.strip('\r').strip('\n')
                    line_list = line.split('\t')
                    
                    if '#' in line_list[1]:
                        continue
                    if line_list[0].split('.')[1] != 'jpg':
                        print(index, line)
                    if len(line_list[-1]) <= 0:
                        continue
                    

                    
                    out_file.write('{}\t'.format(line_list[0]))
                    for char in line_list[-1][:len(line_list[-1]) - 1]:
                        out_file.write('{} '.format(char_to_index[char]))
                    out_file.write('{}\n'.format(char_to_index[line_list[-1][-1]]))

## Test :

    src_test_file_path = 'tmp_data/recognizer_txts/test.txt'
    dst_test_file_path = 'tmp_data/recognizer_txts/real_test.txt'

    with open(dst_test_file_path, 'a') as out_file:
            with open(src_test_file_path, 'r', encoding='utf-8') as in_file:
                lines = in_file.readlines()
                for index, line in enumerate(lines):
                    line = line.strip('\r').strip('\n')
                    line_list = line.split('\t')
                    
                    if '#' in line_list[1]:
                        continue
                    if line_list[0].split('.')[1] != 'jpg':
                        print(index, line)
                    if len(line_list[-1]) <= 0:
                        continue
                    

                    
                    out_file.write('{}\t'.format(line_list[0]))
                    for char in line_list[-1][:len(line_list[-1]) - 1]:
                        out_file.write('{} '.format(char_to_index[char]))
                    out_file.write('{}\n'.format(char_to_index[line_list[-1][-1]]))
