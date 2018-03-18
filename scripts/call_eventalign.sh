#!/usr/bin/env bash

# This is a script which generates event alignment for a set of fast5 files with embedded fastq files

# extracts fastq from fast5
# nanopolish indexes the fastq with fast5 files
# align to reference
# convert to bam, sort, filter out secondary and unmapped reads, index
# run eventalign


while [[ $# -gt 0 ]]
do
key="$1"

case ${key} in
    -f|--fast5dir)
    FAST5DIR="$2"
    shift # past argument
    shift # past value
    ;;
    -t|--threads)
    THREADS="$2"
    shift # past argument
    shift # past value
    ;;
    -r|--reference)
    REFERENCE="$2"
    shift # past argument
    shift # past value
    ;;
    -o|--output)
    OUTPUT="$2"
    shift # past argument
    shift # past value
    ;;
    --rna)
    RNA=true
    shift # past argument
    ;;

esac
done


echo THREADS  = "${THREADS}"
echo REFERENCE = "${REFERENCE}"
echo OUTPUT = "${OUTPUT}"
echo RNA = "${RNA}"
echo FAST5DIR = "${FAST5DIR}"

echo `nanopolish extract ${FAST5DIR} --output ${OUTPUT}/all_files.fastq --fastq`
echo `nanopolish index -d  ${FAST5DIR} ${OUTPUT}/all_files.fastq`


if [ "$RNA" = true ] ; then
    echo `minimap2 -ax splice -uf -k14 ${REFERENCE} ${OUTPUT}/all_files.fastq > ${OUTPUT}/all_files.sam`
else
    echo `bwa mem -x ont2d -t ${THREADS} ${REFERENCE} ${OUTPUT}/all_files.fastq > ${OUTPUT}/all_files.sam`
fi

echo `samtools view -S -b ${OUTPUT}/all_files.sam > ${OUTPUT}/all_files.bam`
echo `samtools sort ${OUTPUT}/all_files.bam -o ${OUTPUT}/all_files.sorted.bam`
# filters out secondary mapped and unmapped reads
# https://www.biostars.org/p/206396/
# https://gist.github.com/davfre/8596159
echo `samtools view -F 260 -F 0x900 -b ${OUTPUT}/all_files.sorted.bam > ${OUTPUT}/unique.sorted.bam`
echo `samtools index ${OUTPUT}/unique.sorted.bam`

echo `nanopolish eventalign --reads ${OUTPUT}/all_files.fastq --bam ${OUTPUT}/unique.sorted.bam --genome ${REFERENCE} > ${OUTPUT}/eventalign.txt`
