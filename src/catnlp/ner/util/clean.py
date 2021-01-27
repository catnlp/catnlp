#!/usr/bin/python3
# -*- coding: utf-8 -*-

def strQ2B(ustr):
    """
    全角转半角
    Args:
        ustr(str): 任意字符串
    Returns:
        rstring(str): 半角字符串
    """
    rstring = ""
    for uchar in ustr:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def clean_text(text):
    """
    清理文本
    Args:
        text(str): 任意文本
    Returns:
        text(str): 清理文本
    """
    text = strQ2B(text)
    text = text.replace("\t", " ")
    return text


if __name__ == "__main__":
    text = "Ａｎｇｅｌａｂａｂｙ    a"
    print(text)
    text = clean_text(text)
    print(text)
