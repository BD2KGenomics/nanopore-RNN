#!/usr/bin/env bash
# This is a script which aligns fasta files to a reference, converts to bam, sorts
# then filters out secondary and unmapped reads

while [[ $# -gt 0 ]]
do
key="$1"

case ${key} in
    -f|--fasta)
    FASTA="$2"
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


echo FASTA  = "${FASTA}"
echo THREADS  = "${THREADS}"
echo REFERENCE = "${REFERENCE}"
echo OUTPUT = "${OUTPUT}"
echo RNA = "${RNA}"


#echo `$FASTA `
if [ "$RNA" = true ] ; then
    echo `minimap2 -ax splice -uf -k14 ${REFERENCE} ${FASTA} > ${OUTPUT}/all_files.sam`
else
    echo `bwa mem -x ont2d -t ${THREADS} ${REFERENCE} ${FASTA} > ${OUTPUT}/all_files.sam`
fi

echo `samtools view -bS ${OUTPUT}/all_files.sam > ${OUTPUT}/all_files.bam`
echo `samtools sort ${OUTPUT}/all_files.bam -o ${OUTPUT}/all_files.sorted.bam`
# filters out secondary mapped and unmapped reads
# https://www.biostars.org/p/206396/
echo `samtools view -F 260 -b ${OUTPUT}/all_files.sorted.bam > ${OUTPUT}/unique.sorted.bam`
echo `jsa.hts.errorAnalysis --bamFile ${OUTPUT}/unique.sorted.bam --reference ${REFERENCE}`



## these are notes for partitioning reads based on mapping to certain references

#poretools fasta 0/ > 0.fasta
#minimap2 -ax splice --splice-flank=no rna_fasta/SIRV_isoforms_multi-fasta_170612a.fasta 0.fasta > 0.SIRV.align.sam
#samtools view -Sq 1 0.SIRV.align.sam > 0.SIRV.filtered.sam
#cat *.SIRV.filtered* > all.SIRV.filtered.sam
#less all.SIRV.filtered.sam | wc -l
#
#
#nanoraw genome_resquiggle 0/ GRCh38_transcriptome/GRCh38_latest_rna.fna --failed-reads-filename 0.failed.reads.fa --bwa-mem-executable  bwa --processes 8
#nanoraw genome_resquiggle 0/ GRCh38_transcriptome/GRCh38_latest_rna.fna --failed-reads-filename 0.failed.reads.fa --graphmap-executable ~/src/graphmap/bin/Linux-x64/graphmap --processes 8
#
#
#
#
#
#nanoraw genome_resquiggle ~/CLionProjects/nanopore-RNN/test_files/minion-reads/rna_reads/ /Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/fake_rnaref/fake_rna.fa --bwa-mem-executable  bwa --processes 2 --overwrite
#
#
## human genome
#minimap2 -ax splice -uf -k14 /home/ubuntu/rna_data/GRCh38_transcriptome/GRCh38_latest_rna.fna 0.fasta > 0.GRCh38.sam
##samtools view -Sq 1 0.GRCh38.sam > 0.GRCh38.filtered.sam
#samtools view -F 260 all.ensemble.forward.sam > unique.mapped.sam
##cat *.SIRV.filtered* > all.SIRV.filtered.sam
#
#
#
