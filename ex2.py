import random

boy = 0
girl = 0

def isGirl():
	if random.random() <= 0.5:
		return True
	else:
		return False

def birth():
    global boy, girl;
    if isGirl():
        girl = girl + 1
        birth()
    else:
        boy = boy + 1

num_boys = 0
num_girls = 0

for y in range(0, 10):
    for x in range(0, 1000000):
        birth()

    print (girl/boy)
    num_boys = num_boys + boy
    num_girls = num_girls + girl
    boy = 0
    girl = 0

print ('Ratio: ' + str(num_girls/num_boys))
