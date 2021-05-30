import re


def cut(line, max_len, overlap_len):
    print(line)
    texts = re.split(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', line)
    texts_len = len(texts)
    print('split:', texts)
    end_idx_list = list()
    texts_len = len(texts)
    i = 0
    end_idx = 0
    result = list()
    while i < texts_len:
        print(i)
        text = texts[i]
        text_len = len(text)
        end_idx += text_len
        print(end_idx)
        if re.match(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', text):
            i += 1
            continue
        # 搜索前缀
        pre_list = list()
        pre_list_len = 0
        j = i - 1
        while j >= 0:
            text_j = texts[j]
            text_j_len = len(text_j)
            if pre_list_len + text_j_len < overlap_len:
                pre_list.append(text_j)
                pre_list_len += text_j_len
            else:
                break
            j -= 1
        pre_idx = 0
        pre_list = pre_list[::-1]
        for tmp_text in pre_list:
            if re.match(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', tmp_text):
                pre_idx += 1
                pre_list_len -= len(tmp_text)
            else:
                break
        pre_list = pre_list[pre_idx:]

        # 搜索后缀
        post_list = list()
        post_list_len = 0
        j = i + 1
        while j < texts_len:
            text_j = texts[j]
            text_j_len = len(text_j)
            if pre_list_len + text_len + post_list_len < max_len:
                post_list.append(text_j)
                post_list_len += text_j_len
            else:
                break
            j += 1
        
        # 拼接
        print('pre:', pre_list)
        print('text', text)
        print('post:', post_list)
        result.append("".join(pre_list + [text] + post_list))
        end_idx += post_list_len
        end_idx_list.append(end_idx)
        i = j


    print('result: ', result)
    print('end: ', end_idx_list)
    print('test')
    for end, text in zip(end_idx_list, result):
        text_len = len(text)
        start = end - text_len
        if line[start: end] != text:
            print("---")
            print(line[start: end])
            print(text)

# def cut(line, max_len, overlap_len):
#     print(line)
#     in_list = list()
#     texts = re.split(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', line)
#     print('split:', texts)
#     start_idx_list = list()
#     start_idx = 0
#     count_in = 0
#     texts_len = len(texts)
#     i = 0
#     end_idx = 0
#     result = list()
#     for text in texts:
#         print(f'inlist: {in_list}')
#         print(f"{i}/{texts_len}")
#         i += 1
#         text_len = len(text)
#         end_idx += text_len
#         if count_in == 0 and re.match(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', text):
#             continue
#         if count_in + text_len < max_len:
#             in_list.append(text)
#             count_in += text_len
#         elif count_in + text_len >= max_len:
#             print("--------")
#             # tmp_idx = 0
#             # for tmp_text in in_list:
#             #     if re.match(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', tmp_text):
#             #         tmp_idx += 1
#             #         count_in -= len(tmp_text)
#             #     else:
#             #         break
#             # in_list = in_list[tmp_idx:]
#             print(in_list)
#             if in_list:
#                 sent = "".join(in_list)
#                 result.append(sent)
#                 start_idx = end_idx - text_len - len(sent)
#                 start_idx_list.append(start_idx)
            
#             if re.match(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', text):
#                 tmp_idx = 0
#                 count_in -= len(in_list[0])
#                 for tmp_text in in_list[1:]:
#                     if re.match(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', tmp_text):
#                         tmp_idx += 1
#                         count_in -= len(tmp_text)
#                     else:
#                         break
#                 in_list = in_list[tmp_idx+1:]
#                 if in_list:
#                     in_list.append(text)
#                     count_in += len(text)
#                 print('continue', in_list)
#                 continue


#             count_tmp = 0
#             tmp_list = [text]
#             while in_list:
#                 tmp_text = in_list.pop()
#                 if count_tmp < overlap_len and text_len + count_tmp < max_len:
#                     tmp_list.append(tmp_text)
#                     count_tmp += len(tmp_text)
#                 else:
#                     in_list = list()
            
#             tmp_idx = len(tmp_list)
#             print('tmp_list: ', tmp_list)
#             for tmp_text in tmp_list[::-1]:
#                 if re.match(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', tmp_text):
#                     tmp_idx -= 1
#                 else:
#                     break
#             tmp_list = tmp_list[: tmp_idx]
#             print('tmp_list: ', tmp_list)
#             count_in = 0
#             while tmp_list:
#                 tmp_text = tmp_list.pop()
#                 in_list.append(tmp_text)
#                 count_in += len(tmp_text)
#             print(in_list)
#             print("+++++++++++++++++")
#     if in_list:
#         sent = "".join(in_list)
#         result.append(sent)
#         start_idx = end_idx - len(sent)
#         start_idx_list.append(start_idx)
#     print('result: ', result)
#     print('start: ', start_idx_list)
#     print('test')
#     for start, text in zip(start_idx_list, result):
#         text_len = len(text)
#         end = start + text_len
#         if line[start: end] != text:
#             print(start, text)


if __name__ == "__main__":
    text = "sd,fsdfdsfdsfdsfdfdsfk,js,sd,kfj,sd,fkdsf,sdfk,sld,jfk,ds,sd,kfs,jdk,f,dfd,dsfs,sdfsdf,sdfsdf"
    cut(text, 20, 8)