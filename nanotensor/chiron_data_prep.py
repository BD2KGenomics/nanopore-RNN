#!/usr/bin/env python
"""Module Docstrings"""
########################################################################
# File: alignment.py
#  executable: alignment.py
#
# Authors: Andrew Bailey
# History: 08/25/17 Created
########################################################################

from __future__ import print_function
import sys
import os
import subprocess
import collections
import re
from timeit import default_timer as timer
from nanonet.fast5 import Fast5
from nanotensor.utils import list_dir, DotDict

alignment_stats = collections.namedtuple('alignment_stats', ['total_reads', 'unaligned_reads', 'deletion_rate',
                                                             'insertion_rate', 'mismatch_rate', 'identity_rate'])


def align_to_reference(sequence, reference, out_bam, threads=1):
    """Align sequence (fasta or fastq) to reference using bwa and samtools"""
    assert os.path.isfile(reference) is True, "reference sequence does not exist"
    # assert os.path.isfile(sequence) is True, "sequence file does not exist"
    bwa_index_genome(reference)
    bwa_command = ["bwa", "mem", "-x", "ont2d", '-t', str(threads), reference, sequence]
    samtools_command = ['samtools', 'view', '-bS']
    # subprocess.call(command)
    p1 = subprocess.Popen(bwa_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2 = subprocess.Popen(samtools_command, stdin=p1.stdout, stdout=open(out_bam, "wb"))
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
    # output = p2.communicate()[0]
    # print(output)
    assert os.path.isfile(out_bam) is True, "Bam file was not created"
    return out_bam


def cat_files(files, out_path):
    """Cat files together into a single file"""
    assert type(files) is list, "Files have to be in a list"
    cat_command = ["cat"]
    cat_command.extend(files)
    subprocess.Popen(cat_command, stdout=open(out_path, "w+"), stderr=subprocess.PIPE)
    assert os.path.isfile(out_path) is True, "Subprocess call didn't work. Make sure regex gives full path"
    return out_path


def get_summary_alignment_stats(bam, reference, report_all=False):
    """Get summary alignment statistics from alignment info using jsa.hts.errorAnalysis"""
    assert os.path.isfile(reference) is True, "reference sequence does not exist"
    assert os.path.isfile(bam) is True, "bam file does not exist"
    command = ["jsa.hts.errorAnalysis", "--bamFile", bam, '--reference', reference]
    p1 = subprocess.Popen(command, stdin=None, stdout=subprocess.PIPE)
    out, err = p1.communicate()
    if report_all:
        alignment_data = out
    else:
        data = re.split('[:\n]', out)
        alignment_data = alignment_stats(total_reads=float(data[11]), unaligned_reads=float(data[13]), deletion_rate=float(data[15]),
                                     insertion_rate=float(data[17]),
                                     mismatch_rate=float(data[19]), identity_rate=float(data[21]))
    return alignment_data


def bwa_index_genome(reference_fasta):
    """Index genome of a given reference fasta file"""
    assert os.path.isfile(reference_fasta), "Reference genome does not exist: {}".format(reference_fasta)
    indexed = check_indexed_reference(reference_fasta)
    if not indexed:
        command = ["bwa", "index", reference_fasta]
        subprocess.call(command)
        indexed = check_indexed_reference(reference_fasta)
    assert indexed is True, "Subprocess call didn't work. Make sure bwa is in the PATH"
    return indexed


# jsa.hts.errorAnalysis --bamFile=test_fastq.bam --reference=reference-sequences/ecoli_k12_mg1655.fa
# samtools view -S -b sample.sam > sample.bam
# bwa mem [options] ref.fa in.fq | samtools view -bS - > out.bam
# bwa index ecoli_k12_mg1655.fa
# bwa mem -x ont2d reference-sequences/ecoli_k12_mg1655.fa test_minion.fa | samtools view -bS - > out.bam

def create_label_file(fast5_object, output_dir, name):
    """Create .signal files from fast5 files for chiron"""
    assert os.path.isdir(output_dir) is True, "output directory does not exist"
    output = os.path.join(output_dir, name + ".label")
    try:
        nanoraw_events, corr_start_rel_to_raw = fast5_object.get_corrected_events()
        events = nanoraw_events["start", 'length', 'base']
        events["start"] = events["start"] + corr_start_rel_to_raw
        with open(output, 'w+') as fh:
            for event in events:
                line = str(event['start']) + ' ' + str(event['start'] + event['length']) + ' ' + str(
                    event['base'] + '\n')
                fh.write(line)
    except KeyError:
        output = False
    return output


def create_signal_file(fast5_object, output_dir, name):
    """Create .label files from fast5 files for chiron"""
    assert os.path.isdir(output_dir) is True, "output directory does not exist"
    output = os.path.join(output_dir, name + ".signal")
    data = fast5_object.get_reads(raw=True, scale=False)
    data1 = next(data)
    with open(output, 'w+') as fh:
        fh.write(' '.join(str(val) for val in data1))

    return output


def label_chiron_data_multiprocess_wrapper(args):
    """Wrapper for label_chiron_data method in order for multiprocessing to be utilized easily"""
    args = DotDict(args)
    # print(args.fast5_path, args.output_dir, args.name)
    signal_path, label_path = label_chiron_data(args.fast5_path, args.output_dir, args.name)
    if args.verbose:
        print("SAVED: {} / {}".format(signal_path, label_path), file=sys.stderr)
    return signal_path, label_path


def label_chiron_data(fast5_path, output_dir, name):
    """Create signal and label data for chiron"""
    fast5_handle = Fast5(fast5_path)
    label_path = create_label_file(fast5_handle, output_dir, name)
    if label_path:
        signal_path = create_signal_file(fast5_handle, output_dir, name)
    else:
        signal_path = False
    return signal_path, label_path


def create_label_chiron_data_args(fast5dir, output_dir, output_name, verbose=False):
    """Create arguments for label_chiron_data function"""
    assert os.path.isdir(fast5dir) is True, "fast5 directory does not exist"
    assert os.path.isdir(output_dir) is True, "output directory does not exist"
    fast5files = list_dir(fast5dir, ext="fast5")
    counter = 0
    for read in fast5files:
        name = output_name + str(counter)
        counter += 1
        yield dict(fast5_path=read, output_dir=output_dir, name=name, verbose=verbose)


def call_nanoraw(fast5dir, reference, num_cpu, overwrite=False):
    """Call nanoraw to label fast5 files with new aligned signal"""
    assert os.path.isdir(fast5dir) is True, "fast5 directory does not exist"
    assert os.path.isfile(reference) is True, "reference fasta file does not exist"
    bwa_index_genome(reference)
    num_cpu = str(num_cpu)
    command = ["nanoraw", "genome_resquiggle", fast5dir, reference, "--bwa-mem-executable", "bwa", "--processes",
               num_cpu]
    if overwrite:
        command.append("--overwrite")
    subprocess.call(command)
    return fast5dir


def check_indexed_reference(reference_fasta):
    """Check to see if the reference genome has be inedexed by bwa"""
    assert os.path.isfile(reference_fasta), "Reference genome does not exist: {}".format(reference_fasta)
    exts = ["amb", "bwt", "pac", "sa", "ann"]
    indexed = True
    for ext in exts:
        if not os.path.isfile(reference_fasta + '.' + ext):
            indexed = False
    return indexed


def main():
    """Main docstring"""
    start = timer()
    ont_fasta = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/test_minion.fa"
    ont_fastq = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/test_minion.fastq"
    test_fast5 = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_mux_scan_AMS_158_R9_WGA_Ecoli_08_20_16_83098_ch467_read35_strand.fast5"

    chiron_fast5_dir = "/Users/andrewbailey/CLionProjects/nanopore-RNN/methylated_test"
    ecoli_genome = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/reference-sequences/ecoli_k12_mg1655.fa"

    fasta_dir = "/Users/andrewbailey/CLionProjects/nanopore-RNN/chiron/test_output/result"
    # files = list_dir(fasta_dir, ext="fasta")
    # output = "/Users/andrewbailey/CLionProjects/nanopore-RNN/chiron/test_output/result/all_reads2.fasta"
    # print(files)
    # path = cat_files(files, output)
    # bam = align_to_reference(path, ecoli_genome,
    #                          "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/test3.bam", threads=2)
    # data = get_summary_alignment_stats(bam, ecoli_genome, report_all=True)
    # print(data)
    # call_nanoraw("/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/canonical", ecoli_genome, 1)
    # for x in range(len(data)):
    #     print(x, (data[x]))

    fast5 = Fast5(test_fast5)
    create_label_file(fast5, "/Users/andrewbailey/CLionProjects/nanopore-RNN/", "test1")
    # call_nanoraw(chiron_fast5_dir, ecoli_genome, 2, overwrite=True)
    # indexed = check_indexed_reference(ecoli_genome)
    # print(indexed)
    #
    # if not indexed:
    #     bwa_index_genome(ecoli_genome)
    #     indexed = check_indexed_reference(ecoli_genome)
    #
    # print(indexed)
    # fast5 = Fast5(test_fast5)
    # # data = fast5.get_reads(raw=True, scale=False)
    # # data1 = (next(data))
    # # print(data1)
    # # with open("/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/ch467_read35.signal", 'w+') as file:
    # #     for x in data1:
    # #         file.write(str(x)+' ')
    # nanoraw_events = fast5.get_corrected_events()
    # events = nanoraw_events["start", 'length', 'base']
    # with open("/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/ch467_read35.label", 'w+') as file:
    #     for event in events:
    #         line = str(event['start']) + ' ' + str(event['start'] + event['length']) + ' ' + str(event['base'] + '\n')
    #         file.write(line)


    # print(row1)
    # print(row1[["start", 'length', 'base']])
    # print(row1['start'])
    #
    # print(row1['length'])
    # print(row1['base'])

    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
