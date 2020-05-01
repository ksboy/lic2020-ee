#coding=utf-8
import sys 
#str = input()
def helper(aList, left, right):
    pivot = aList[start]
    while left < right:
        while left < right and nums[right]>= pivot:
            right -= 1
        aList[left]= aList[right]
        while left < right and nums[right]<= pivot:
            left += 1
        aList[right] = aList[left]
    aList[left] = pivot
    return left
    
def quickSort(nums, left , right):
    if left<right:
        index = helper(nums ,left, right)
        quickSort(nums, left, index-1)
        quickSort(nums, index+1, right)
    return nums

quickSort([4,3,2,1],0,3)
    
