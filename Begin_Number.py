
# %%
import numpy as np
def get_beginning_num(one_try):
		''' Find the beginning number of the longest accending subset 
		in a series of integers

		INPUT
		------
			intput_list: list
				a list of integers whose length is unknown

		OUTPUT
		------
			beginning_num: int
				the beginning number(s) of the longest accending subset
		'''
		length = len(one_try)
		max_l = 1
		current_l = 1
		fois = 1
		beginning_num = ['']*length
		for i in range(length):
			if one_try[i] > one_try[i-1]:
				current_l+=1
				#print(current_l)
				if i == length-1:
					if current_l > max_l:
						max_l = current_l
						print(max_l)
						fois = 1
						beginning_num[fois-1] = one_try[i-max_l+1]
						current_l = 1
					if current_l == max_l:
						fois += 1
						beginning_num[fois-1] = one_try[i-max_l+1]
						current_l =1
			else:
				if current_l > max_l:
					max_l = current_l
					print(max_l)
					beginning_num[fois-1]= one_try[i-max_l]
					fois = 1
					current_l = 1
				if current_l == max_l:
					fois += 1
					beginning_num[fois-1] = one_try[i-max_l]
					current_l =1
		return beginning_num
haha = [3,7,6,4,5,7,8,2,6,8,9,10,11,1,2,3,4,5,6,4,22,23,25,26,34]
get_beginning_num(haha)

# %%	


# %%
