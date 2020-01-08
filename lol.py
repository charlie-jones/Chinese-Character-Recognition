def isInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

f = open("images/10tens.txt", 'r')
f2 = open("images/10tens new.txt",'w+')
filedata = f.readlines()
f.close()
data = []

for line in filedata:
    i = 0
    line = line.split()
    for w in line:
        if isInt(w) == True:
            if int(w) < 155:
                line[i] = w.replace(w, '0')
            else:
                line[i] = w.replace(w, '255')
        i+=1
    f2.write(" ".join(line) + "\n")


f2.close()


print("done")

