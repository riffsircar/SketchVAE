import sys
import os
import random
import json

levels = []
i = 0
dims = (11,16)

for file in os.listdir('sketches/'):
    print(file)
    data = open('sketches/' + file,'r').read().splitlines()
    data = [line.replace('\r\n','') for line in data]
    for h_offset in range(len(data)-dims[0]+1):
        for w_offset in range(len(data[0])-(dims[1]-1)):
            write = True
            out = []
            for line in data[h_offset:h_offset+dims[0]]:
                out.append(line[w_offset:w_offset+dims[1]])
            if any('@' in line for line in out):
                write = False

            if write:
                outfile = open('chunks/cv_chunk_' + str(i) + '.txt','w')
                for (_,line) in enumerate(out):
                    outfile.write(line + '\n')
                outfile.close()
                i += 1
