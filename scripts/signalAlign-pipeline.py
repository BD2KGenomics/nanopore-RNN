#!/usr/bin/env python
"""
1) Creating a Bed file:

2) pre-processing the refrence sequence by removing "\n" and the > line

3) creating a BED file which replaces the second C in CC(A/T)GG motif to E for temp and comp

"""
import glob
import random
import numpy as np





ref = open("E.coli_K12.fasta", "r")
ref_edited = open("E.coli_K12-modified.fasta", "w")

for i in ref.readlines():
    if ">" in i:
        ref_edited.write(i)
    else:
        ref_edited.write(i.split("\n")[0])
ref.close()
ref_edited.close()


ref = open("E.coli_K12-modified.fasta", "r")
ref_edited = open("no-N.txt", "w")
for i in ref.readlines():
    if ">" not in i:
        ref_edited.write(i)
ref.close()
ref_edited.close()




# Bed file C=>E
# + strand
ref_edited = open("no-N.txt", "r")
seq_string = ''.join(ref_edited)
output = open("C-to-E.bed", "w")
for i in range(len(seq_string)):
    if seq_string.startswith('CCAGG', i):
        output.write("gi_ecoli"+ "\t" + np.str(i + 1) + "\t" + "+" + "\t" + "C" +"\t" + "E" + "\n")
    elif seq_string.startswith('CCTGG', i):
        output.write("gi_ecoli"+ "\t" + np.str(i + 1) + "\t" + "+" + "\t" + "C" +"\t" + "E" + "\n")
output.close()
ref_edited.close()


# In[114]:

# - strand
opening = open("C-to-E.bed", "a")

for i in range(len(seq_string)):
    if seq_string.startswith('GGTCC', i):
        opening.write("gi_ecoli"+ "\t" + np.str(i + 1) + "\t" + "-" + "\t" + "C" +"\t" + "E" + "\n")
    elif seq_string.startswith('GGACC', i):
        opening.write("gi_ecoli"+ "\t" + np.str(i + 1) + "\t" + "-" + "\t" + "C" +"\t" + "E" + "\n")
opening.close()


# # 2.a Shell command for running signalAlign on gEoli
#

# runSignalAlign -d 08_05_16_R9_gEcoli_2D_500 -r E.coli_K12.fasta-modified.fasta -o 08_05_16_R9_gEcoli_2D_500-op-assign/ -f assignments -T ~/nanonet-scott/signalAlign/models/testModelR9_5mer_acegt_template.model -C ~/nanonet-scott/signalAlign/models/testModelR9_5mer_acegt_complement.model -p C-to-E.bed --2d
# *note threshold can be set to 0 with this flag -t = 0

# # 2.b Shell command for running signalAlign on WGA

# runSignalAlign -d 08_20_16_R9_WGA_Ecoli_500 -r E.coli_K12.fasta-modified.fasta -o 08_20_16_R9_WGA_Ecoli_500-op-assign/ -f assignments -T ~/nanonet-scott/signalAlign/models/testModelR9_5mer_acegt_template.model -C ~/nanonet-scott/signalAlign/models/testModelR9_5mer_acegt_complement.model --2d
# *note threshold can be set to 0 with this flag -t = 0

# # 3.a  Concatenate assignments into one single assignment


read_files = glob.glob("Path-to-WGA-assignments/*.assignments")

with open("WGA-assignments/WGA-assignments.tsv", "w") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())

read_files = glob.glob("Path-to-gEcoli-assignments/*.assignments")

with open("gEcoli-assignments/gEcoli-assignments.tsv", "w") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())

filenames = ["gEcoli-assignments/gEcoli-assignments.tsv", "WGA-assignments/WGA-assignments.tsv"]
with open("allAssignments-concatenated/all-assign.tsv", "w") as output:
     for File in filenames:
             with open(File) as Input:
                     for line in Input:
                             output.write(line)


# # 3.b  get 1-50 assignments for each kmer


#creating a dictionary
kmerDict = dict()
opening= open("/Volumes/Rojin/nanopore-project/allAssignments-concatenated/all-assign.tsv", "r")
for i in opening.readlines():
    key = i.split("\t")[0]
    value = "\t".join(i.split("\t")[1:])
    if kmerDict.has_key(key):
        kmerDict[key].append(value)
    else:
        kmerDict[key] = [value]
opening.close()

#sampling
opening = open("/Volumes/Rojin/nanopore-project/allAssignments-concatenated/50-ofEachKmer.tsv", "w")
for key,value in kmerDict.iteritems():
    mylist = kmerDict[key]
    if len(mylist) >= 50:
        rand_smpl = [ mylist[i] for i in random.sample(range(len(mylist)), 50)]
        for g in rand_smpl:
            string = ''.join(g)
            opening.write(key + "\t" + string)
    elif len(mylist) <50:
        rand_smpl = [ mylist[i] for i in random.sample(range(len(mylist)), len(mylist))]
        for g in rand_smpl:
            string = ''.join(g)
            opening.write(key + "\t" + string)
opening.close()


# # 3.c Run buildHdpUtil to build an hdp model

# buildHdpUtil -T ~/nanonet-scott/signalAlign/models/testModelR9_5mer_acegt_template.model -C ~/nanonet-scott/signalAlign/models/testModelR9_5mer_acegt_complement.model -v hdp-build-temp/hdp-temp.hdp -w hdp-build-comp/hdp-comp.hdp -l allAssignments-concatenated/50-ofEachKmer.tsv --verbose -p 10 -a 5 -n 15000 -I 30 -t 100 -s 50 -e 140 -k 1800 -g 1 -r 1 -j 1 -y 1 -i 1 -u 1


# # 4. create a BED file C->X

# Bed file C=>X
# + strand
ref_edited = open("../GGG/test_sequences/no-N.txt", "r")
seq_string = ''.join(ref_edited)
output = open("../GGG/test_sequences/C-to-X.bed", "w")
for i in range(len(seq_string)):
    if seq_string.startswith('CCAGG', i):
        output.write("gi_ecoli"+ "\t" + np.str(i + 1) + "\t" + "+" + "\t" + "C" +"\t" + "X" + "\n")
    elif seq_string.startswith('CCTGG', i):
        output.write("gi_ecoli"+ "\t" + np.str(i + 1) + "\t" + "+" + "\t" + "C" +"\t" + "X" + "\n")
output.close()
ref_edited.close()

# - strand
opening = open("../GGG/test_sequences/C-to-X.bed", "a")

for i in range(len(seq_string)):
    if seq_string.startswith('GGTCC', i):
        opening.write("gi_ecoli"+ "\t" + np.str(i + 1) + "\t" + "-" + "\t" + "C" +"\t" + "X" + "\n")
    elif seq_string.startswith('GGACC', i):
        opening.write("gi_ecoli"+ "\t" + np.str(i + 1) + "\t" + "-" + "\t" + "C" +"\t" + "X" + "\n")
opening.close()


# # 5. run signalAlign again with
# ## 1)hdp model, 2) hmm model, 3) c->x bed file

# runSignalAlign -d 08_05_16_R9_gEcoli_2D_500 -r E.coli_K12.fasta-modified.fasta -o SA-final/ -p C-to-X.bed -f full -x cytosine2 -T ~/nanonet-scott/signalAlign/models/testModelR9_5mer_acegt_template.model -C ~/nanonet-scott/signalAlign/models/testModelR9_5mer_acegt_complement.model -tH hdp-build-temp/hdp-temp.hdp -cH hdp-build-comp/hdp-comp.hdp -smt=threeStateHdp --2d
