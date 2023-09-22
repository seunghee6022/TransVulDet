import json

# add artifical root to make 0 class validate
# 0 - 10000-0 
# 1000 - 10000-1000
data_str = """
0 - 0
20 - 1000-707-20
22 - 1000-664-706-22
23 - 1000-664-706-22-23
27 - 1000-664-706-22-23-27
59 - 1000-664-706-59
73 - 1000-664-669-642-73, 1000-664-610-73
74 - 1000-707-74
77 - 1000-707-74-77
78 - 1000-707-74-77-78
79 - 1000-707-74-79
80 - 1000-707-74-79-80
87 - 1000-707-74-79-87
88 - 1000-707-74-77-78
89 - 1000-707-74-943-89
90 - 1000-707-74-943-90
91 - 1000-707-74-91
93 - 1000-707-74-93
94 - 1000-707-74-94, 1000-664-913-94
98 - 1000-664-706-98, 1000-664-669-829-98
113 - 1000-707-74-93-113, 1000-435-436-113
114 - 1000-664-669-642-73-114, 1000-664-610-73-114
116 - 1000-707-116
117 - 1000-707-116-117
119 - 1000-664-118-119
120 - 1000-664-118-119-120
121 - 1000-664-118-119-787-121, 1000-664-118-119-788-121
122 - 1000-664-118-119-787-122, 1000-664-118-119-788-122
125 - 1000-664-118-119-125
126 - 1000-664-118-119-125-126, 1000-664-118-119-788-126
129 - 1000-707-20-1285-129
131 - 1000-682-131
134 - 1000-664-668-134
138 - 1000-707-138
170 - 1000-707-170
172 - 1000-707-172
178 - 1000-664-706-178
184 - 1000-693-184, 1000-697-1023-184
185 - 1000-697-185
187 - 1000-697-1023-187
190 - 1000-682-190
191 - 1000-682-191
193 - 1000-682-193
200 - 1000-664-668-200
203 - 1000-664-668-200-203
208 - 1000-664-668-200-203-208
209 - 1000-664-668-200-209, 1000-703-755-209
212 - 1000-664-669-221
221 - 1000-664-221
241 - 1000-703-228-241
248 - 1000-703-755-248, 1000-691-705-248
250 - 1000-710-657-250 
252 - 1000-703-754-252
266 - 1000-284-269-266
269 - 1000-284-269
273 - 1000-284-269-271-273, 1000-703-754-273 
276 - 1000-664-668-732-276
277 - 1000-664-668-732-277
279 - 1000-664-668-732-279
281 - 1000-664-668-732-281
284 - 1000-284
285 - 1000-284-285
287 - 1000-284-287
288 - 1000-284-287-306-288
290 - 1000-284-287-1390-290
294 - 1000-284-287-1390-294
295 - 1000-284-287-295
297 - 1000-284-287-295-297
303 - 1000-284-287-1390-303
304 - 1000-284-287-1390-303-304, 1000-710-573-304
305 - 1000-284-287-1390-305
306 - 1000-284-287-306
307 - 1000-691-799-307, 1000-284-287-1390-307
311 - 1000-693-311
312 - 1000-693-311-312
319 - 1000-693-311-319
321 - 1000-693-330-344-798-321
324 - 1000-664-666-672-324
326 - 1000-693-326
327 - 1000-693-327
328 - 1000-693-326-328, 1000-693-327-328
330 - 1000-693-330
331 - 1000-693-330-331
335 - 1000-693-330-335
338 - 1000-693-330-338
345 - 1000-693-345
346 - 1000-693-345-346
347 - 1000-693-345-347
350 - 1000-693-807-350, 1000-284-287-1390-290-350
352 - 1000-693-345-352
354 - 1000-693-345-354, 1000-703-754-354
358 - 1000-710-573-359, 1000-693-358
359 - 1000-664-668-200-359
362 - 1000-691-362
367 - 1000-691-367
369 - 1000-682-369
378 - 1000-664-668-377-378
379 - 1000-664-668-377-379
384 - 1000-664-610-384
400 - 1000-664-400
401 - 1000-664-404-772-401
404 - 1000-664-404
415 - 1000-664-666-415
416 - 1000-664-118-119-825-416
425 - 1000-284-287-306-288-425, 1000-284-285-862-425
426 - 1000-664-673-426, 1000-664-668-642-426
427 - 1000-664-668-427
428 - 1000-664-668-428
434 - 1000-664-669-434
436 - 1000-435-436
441 - 1000-664-610-441
444 - 1000-435-436-444
457 - 1000-664-665-908-457
459 - 1000-664-404-459
467 - 1000-682-131-467
469 - 1000-682-469
470 - 1000-664-610-470, 1000-664-913-470
471 - 1000-664-471
475 - 1000-710-573-475
476 - 1000-710-476, 1000-703-754-476
494 - 1000-664-669-494
502 - 1000-664-913-502
506 - 1000-710-684-912-506
521 - 1000-284-287-1390-1391-521
522 - 1000-664-668-522, 1000-284-287-1390-522
527 - 1000-284-285-552-527
532 - 1000-664-668-200-538-532
538 - 1000-664-668-200-538
552 - 1000-284-285-552, 1000-664-668-552
565 - 1000-664-668-642-565, 1000-693-602-565
573 - 1000-710-573
590 - 1000-664-404-763-762-590
601 - 1000-664-610-601
610 - 1000-664-610
611 - 1000-664-610-611
613 - 1000-664-666-672-613
617 - 1000-691-670-617
639 - 1000-664-706-66-639
640 - 1000-284-287-1390-640
641 - 1000-707-74-99-641
644 - 1000-707-116-644
662 - 1000-664-662, 1000-691-662
664 - 1000-664
665 - 1000-664-665
666 - 1000-664-666
667 - 1000-691-662-667
668 - 1000-664-668
669 - 1000-664-669
670 - 1000-691-670
672 - 1000-664-666-672
673 - 1000-664-673
674 - 1000-691-834-674
676 - 1000-710-1177-676
680 - 1000-682-190-680
681 - 1000-664-704-681
682 - 1000-682
684 - 1000-710-684
693 - 1000-693
697 - 1000-697
704 - 1000-664-704
706 - 1000-664-706
707 - 1000-707
732 - 1000-284-285-732, 1000-664-668-732
749 - 1000-284-749
754 - 1000-703-754
755 - 1000-703-755
758 - 1000-710-758 
759 - 1000-693-326-328-916-759, 1000-693-327-328-916-759
763 - 1000-664-404-763
770 - 1000-664-400-770, 1000-664-665-770
772 - 1000-664-404-772
774 - 1000-664-400-770-774
776 - 1000-691-834-674-776, 1000-664-400-405-776
786 - 1000-664-118-119-786
787 - 1000-664-118-119-787
788 - 1000-664-118-119-788
791 - 1000-707-138-790-791
798 - 1000-693-330-344-798
805 - 1000-664-118-119-805
823 - 1000-664-118-119-823
824 - 1000-664-118-119-824
825 - 1000-664-118-119-825, 1000-664-666-672-825
829 - 1000-664-449-829
834 - 1000-691-834
835 - 1000-691-834-835
838 - 1000-707-116-838
843 - 1000-664-704-843
862 - 1000-284-285-862
863 - 1000-284-285-863
908 - 1000-664-665-908
909 - 1000-664-665-909
913 - 1000-664-913
915 - 1000-664-913-915
916 - 1000-666-672-825-916
918 - 1000-664-610-441-918
922 - 1000-664-922
924 - 1000-693-345-924
943 - 1000-707-74-943
1021 - 1000-664-610-441-1021, 1000-664-221-452-1021
1188 - 1000-664-665-1188
1236 - 1000-707-74-1236
1321 - 1000-664-913-915-1321
1333 - 1000-664-400-405-407-1333
1336 - 1000-664-913-914-1336 
"""

# Add new root 10000 to all nodes
data_lines = data_str.strip().split("\n")
new_data_lines = []

for line in data_lines:
    parts = line.split(" - ")
    new_sequences = []
    for sequence in parts[1].split(", "):
        if not sequence.startswith("10000-"):
            sequence = "10000-" + sequence
        new_sequences.append(sequence)
    new_line = parts[0] + " - " + ", ".join(new_sequences)
    new_data_lines.append(new_line)

new_data_str = "\n".join(new_data_lines)

# Save it as json file
data_lines = new_data_str.strip().split('\n')

data_dict = {}
for line in data_lines:
    if ' - ' in line:  # Check if the separator exists in the line
        key, value = line.split(' - ')
        # if ', ' in value:
        #     continue
        # data_dict[key] = value
        data_dict[key] = value.split(', ')
    else:
        print(f"Skipping line: {line}")

# Save the dictionary as a JSON file
with open('data_preprocessing/preprocessed_datasets/graph_all_paths.json', 'w') as f:
    json.dump(data_dict, f, indent=4)


