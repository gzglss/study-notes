from eda import *

def gen_aug(ori,out,alpha=0.1,num_aug=9):
    with open(ori,'r') as infile,open(out,'w') as outfile:
        while True:
            line=infile.readline()
            if not line:
                break
            line=line.strip()
            res=eda(line)
            for s in res:
                outfile.write(s+'\n')

        infile.close()
        outfile.close()

gen_aug("../data/query.txt",'../data/gen_query.txt')