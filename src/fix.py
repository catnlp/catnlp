import re
from pathlib import Path


def process(source, target, method):
    with open(source, "r", encoding="utf-8") as sf, \
        open(target, "w", encoding="utf-8") as tf:
        if method == "address":
            address(sf, tf)
        else:
            raise ValueError


def address(sf, tf):
    for line in sf:
        line = line.rstrip()
        if not line:
            continue
        idx, text, tags = line.split("\u0001")
        if re.search(r"^(A+&)?A+$", text):
            tags = " ".join(["O"] * len(text))
            tf.write(f"{idx}\u0001{text}\u0001{tags}\n")
            continue
        tag_list = tags.split()
        poi_tag = False
        i = 0
        tag_len = len(tag_list)
        while i < tag_len:
            tag = tag_list[i]
            if tag[0] in ["B", "S"]:
                tag_name = tag[2:]
                if tag_name not in ["poi", "subpoi"]:
                    i += 1
                    continue
                if poi_tag:
                    break
                # if poi_tag and tag_name == "poi":
                #     if tag[0] == "S":
                #         tag[i] = "S-subpoi"
                #         j = i + 1
                #     else:
                #         tag_list[i] = "B-subpoi"
                #         j = i + 1
                #         while j < tag_len:
                #             if tag_list[j] == "I-poi":
                #                 tag_list[j] = "I-subpoi"
                #                 j += 1
                #             elif tag_list[j] == "E-poi":
                #                 tag_list[j] = "E-subpoi"
                #                 j += 1
                #                 break
                #             else:
                #                 break
                #     i = j
                #     continue
                if not poi_tag and tag_name == "subpoi":
                    if tag[0] == "S":
                        tag_list[i] = "S-poi"
                        j = i + 1
                    else:
                        tag_list[i] = "B-poi"
                        j = i + 1
                        while j < tag_len:
                            if tag_list[j] == "I-subpoi":
                                tag_list[j] = "I-poi"
                                j += 1
                            elif tag_list[j] == "E-subpoi":
                                tag_list[j] = "E-poi"
                                j += 1
                                break
                            else:
                                break
                    poi_tag = True
                    i = j
                    continue
                elif not poi_tag and tag_name == "poi":
                    poi_tag = True
            i += 1
        
        match = re.search(r"((?<![A-Za-z0\-—])(0+(-|&|—))?0+$)|(（|）)|(多处)", text)
        if match:
            match_str = match.group()
            start, end = match.span()
            # if tag_list[start][0] == "B":
            for i in range(start, end):
                tag_list[i] = "O"
        tags = " ".join(tag_list)
        tf.write(f"{idx}\u0001{text}\u0001{tags}\n")
        
        

if __name__ == "__main__":
    source_path = Path("resources/data/dataset/ner/zh/ccks/address/0621")
    source_file = source_path / "试一下_addr_parsing_runid_0628.txt"
    target_file = source_path / "试一下_addr_parsing_runid.txt"
    process(source_file, target_file, method="address")


"""
((0+-)?[A0一二三四五六七八九十]\tO\n)+(楼|单|层|室)\tO\n(元\tO\n)?
"""