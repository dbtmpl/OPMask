"""
Looks ugly, but in my defense:
Zen of Python: Namespaces are one honking great idea -- let's do more of those!
"""

"""
The COCO category Ids of the Pascal VOC categories.
"""
VOC_IDS = (1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72)

"""
Convenience function getting the continuous indices corresponding to the Pascal VOC
categories. With this we can index the correct Pascal VOC categories from a class \
specific tensor.
"""
VOC_INDICES = (0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62)

"""
Convenience function getting the COCO category Ids of the categories that are not part
of the Pascal VOC dataset.
"""
NON_VOC_IDS = (
    8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
)

"""
Convenience function getting the continuous indices corresponding to the categories
that are not part of the Pascal VOC dataset. With this we can index the Non Pascal VOC
categories from a class specific tensor.
"""
NON_VOC_INDICES = (
    7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65, 66,
    67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79
)

"""
VOC_INDICES + 20 random classes from NON_VOC_INDICES
"""
FOURTY_INDICES_INC = (
    0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62, 9, 10, 13, 23, 24, 25,
    26, 36, 42, 47, 48, 49, 52, 54, 61, 63, 70, 74, 76, 79
)
