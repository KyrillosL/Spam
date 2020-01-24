import csv
from os import listdir
from os.path import isfile, join
import numpy as np
columns = [ 'id',
            'word_freq_make',
            'word_freq_address',
            'word_freq_all',
            'word_freq_3d',
            'word_freq_our',
            'word_freq_over',
            'word_freq_remove',
            'word_freq_internet',
            'word_freq_order',
            'word_freq_mail',
            'word_freq_receive',
            'word_freq_will',
            'word_freq_people',
            'word_freq_report',
            'word_freq_addresses',
            'word_freq_free',
            'word_freq_business',
            'word_freq_email',
            'word_freq_you',
            'word_freq_credit',
            'word_freq_your',
            'word_freq_font',
            'word_freq_000',
            'word_freq_money',
            'word_freq_hp',
            'word_freq_hpl',
            'word_freq_george',
            'word_freq_650',
            'word_freq_lab',
            'word_freq_labs',
            'word_freq_telnet',
            'word_freq_857',
            'word_freq_data',
            'word_freq_415',
            'word_freq_85',
            'word_freq_technology',
            'word_freq_1999',
            'word_freq_parts',
            'word_freq_pm',
            'word_freq_direct',
            'word_freq_cs',
            'word_freq_meeting',
            'word_freq_original',
            'word_freq_project',
            'word_freq_re',
            'word_freq_edu',
            'word_freq_table',
            'word_freq_conference',
            'char_freq_;',
            'char_freq_(',
            'char_freq_[',
            'char_freq_!',
            'char_freq_$',
            'char_freq_#',
            'capital_run_length_average',
            'capital_run_length_longest',
            'capital_run_length_total',
            'is_spam'
           ]

print(len(columns))

path = '/Users/Cyril_Musique/Documents/Cours/M2/fouille_de_données/Projet/'
path_file_to_read = '/Users/Cyril_Musique/Documents/Cours/M2/fouille_de_données/Projet/spambase.data'

file_to_read = open(path_file_to_read, 'r')
#Clear the file
path_file_to_write = "/Users/Cyril_Musique/Documents/Cours/M2/fouille_de_données/Projet/dataset.csv"
file_to_write = open(path_file_to_write, "w+")
file_to_write.close()

index = 0
with open(path_file_to_write, 'a') as csvFile:


    writer = csv.writer(csvFile)
    writer.writerow(columns)


    #contents = file_to_read.read()
    #print(contents)
    f = file_to_read.readlines()
    for line in f:
        currentline = line.split(",")
        #print(len(currentline))
        for word in currentline:
            word.replace("\n", "")
        row= [index]+ currentline
        #print(len(row))
        writer.writerow(row)

csvFile.close()
