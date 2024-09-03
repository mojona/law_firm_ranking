"""
This script extracts and clusters the roles and law firms from the attorney strings in the cases_df.
This results in two columns - extracted_roles and extracted_firms - being added to cases_df.
"""

import gzip
import csv
import logging
import ast
import re
import pandas as pd
import numpy  as np

from sklearn.cluster    import AgglomerativeClustering
from rapidfuzz.distance import Levenshtein

from routines           import get_dir


def term_expanded_general(object, number, list_left, list_right):
    '''
    This function extracts a substring from an 'object' (typically a string) based on a given 'number' (index).
    It expands to the left and right of the index as long as the characters are alphabetic or in the provided lists
    'list_left' and 'list_right'. The function returns the extracted substring.

    :param object:                  String from which to extract the substring
    :param number:                  Index from which to start extracting the substring
    :param list_left:               List of non-alphabetic characters allowed on the left side of the substring
    :param list_right:              List of non-alphabetic characters allowed on the right side of the substring
    :return:                        Extracted substring
    '''

    if number < 1 or number >= len(object): return ''                       # case where number is out of bounds
    
    # expanding to the right
    ####################################################################################################################
    iterator = number
    while iterator      < len(object):
        if object[iterator].isalpha() == False and object[iterator] not in list_right:  # check right side of string
            break
        iterator            += 1
    
    # expanding to the left
    ####################################################################################################################
    iterator_neg            = number
    while iterator_neg  > -1:
        if object[iterator_neg - 1].isalpha() == False and object[
            iterator_neg - 1] not in list_left:                                         # check left side of string
            break
        iterator_neg        -= 1
        if iterator_neg == 0:   return object[iterator_neg:iterator]                    # beginning of object reached
        
    return object[iterator_neg:iterator]


def extract_firms_roles(attorney):
    '''
    This function cleans attorney strings and extracts:
    - the  law firms:
        of type A) 'Smith & Johnson' (containing an ' & ')
        or tpye B) 'Smith LLP' (containing a legal abbreviation)
    - the roles: expected to follow a ' for ' sucha as ' for plaintiff'

    :param attorney:        list of attorney strings
    :return:                list of roles, list of corresponding lists of law firms
    '''
    
    # minor helper functions
    ####################################################################################################################

    def find_first_right(object, pos, list=['-']):
        # returns index of the first term to the right of pos, otherwise len(object)
        # do not start with ' '
        i = pos
        while i < len(object):
            if object[i].isalpha() == False and object[i] not in list:
                return i - 1
            i += 1
        return i + 1

    def procede_until_lst_left(object, pos, lst):
        # look for position of terms in lst on the left of pos
        while pos > -1:
            if object[pos] in lst:
                return pos
            if object[pos:pos + 2] in lst:
                return pos + 1
            pos -= 1
        return pos + 1

    def find_upper_after_lower(object, pos):
        # find uppercase term with lowercase term to its left
        def find_lowercase_left(object, pos):
            # returns index of the first lowercase term to the left of pos, otherwise 0
            while pos > -1:
                if (object[pos + 1].islower() == True and object[pos] == ' ') or (
                        object[pos + 1].islower() == True and object[pos] == '('):
                    return pos + 1
                pos -= 1
            return pos + 1
        
        k = find_lowercase_left(object, pos)
        if k == 0 and object[k].isupper() == True:  return 0
        i = k + 1
        while i < len(object):
            if object[i] == ' ':    return i + 1  # returns index of uppercase word
            i += 1
        return i + 1

    def two_upper_no_comma_left(object, pos): 
        # look for two terms uppercase words not separated by comma-->choose word to the right
        def find_first_left(object, pos, list=['-']):
            # returns index of the first term to the left of pos, otherwise 0
            i = pos
            while i > -1:
                if object[i].isalpha() == False and object[i] not in list:
                    return i + 1
                i -= 1
            return i + 1
        while pos > -1:
            if object[pos] == ' ' and object[pos - 1] != ',' and object[
                find_first_left(object, pos - 2)].isupper() == True and object[pos + 1].isupper() == True:
                return find_first_right(object, pos + 2) + 3
            pos -= 1
        return pos + 1
        
    # The following two functions are designed to extract law firms of the following two types:
    # A) law firms of the type Smith & Johnson (including ' & ')
    # B) law firms of the type Smith LLP (including abbreviations such as LLP)
    ####################################################################################################################

    def find_replace_lawfirm_with(object):
        '''
        Takes object, a string, and returns law firm containing ' & ' and the original object without such law firms.
        iter_end indicates beginning of law firm, iter_beg indicates ending of law firm.

        A       look for ' & '
        A.1     navigate to its right
        A.2     if to its right ' & associates': "single expression" to its left: procede until reaching lowercase or 
                ':' ')' '(' ', '
        A.2     navigate to its left:
        A.2.1   if ', ' before first term:
                -->continue to the left until reaching lowercase, ':' ')' '(' or two uppercase without comma:
                -->in latter case do not count term before last comma
        A.2.2   if no ', ' before first term: continue to the left until reaching lowercase, ':' ')' '(' ', '
        A.3     replace law firm with :::

        :param object:          string
        :return:                list of law firms, object without law firms
        '''

        list_of_lawfirms = []
        while ' & ' in object:  # look for ' & '
            iter_end        = find_first_right(object, object.rfind(' & ') + 3) + 1

            if object[procede_until_lst_left(object, object.rfind(' & ') - 2, ' ') - 1] != ',' \
                    or object[object.rfind(' & ') + 3:object.rfind(' & ') + 13] == 'Associates':
                    # case without comma before first term, associates
                iter_obj    = procede_until_lst_left(object, object.rfind(' & ') - 1,
                                                  [': ', '(', ') ', ', ', '; ', '] ', '],']) + 1
                iter_upper  = find_upper_after_lower(object, object.rfind(' & ') - 1)
                iter_beg    = max(iter_obj, iter_upper)

            else:  # case with comma before first term
                iter_obj    = procede_until_lst_left(object, object.rfind(' & ') - 1, \
                                                     [': ', '(', ') ', '; ', '] ', '],']) + 1
                iter_upper  = find_upper_after_lower(object, object.rfind(' & ') - 1)
                iter_2up    = two_upper_no_comma_left(object, object.rfind(' & ') - 1)
                iter_beg    = max(iter_obj, iter_upper, iter_2up)

            if iter_beg     == 1:
                list_of_lawfirms += [object[iter_beg - 1:iter_end]]
                object      = object.replace(object[iter_beg - 1:iter_end + 1], ' ::: ')
            else:
                list_of_lawfirms += [object[iter_beg:iter_end]]
                object      = object.replace(object[iter_beg:iter_end + 1], ' ::: ')

        return [list_of_lawfirms, object]

    def find_replace_lawfirm_abbrev(object):
        '''
        B       look for potential law firm abbrevation name
        B.1     law office of/law offices of --> navigate to the right
        B.1     navigate to its left (potentially preceded by comma)
        B.2     if immediately reaching ':::' stop
                [consider case with 'and' --> replacing it with '&']
        B.3     if ', ' before first term:
                -->continue to the left until reaching lowercase, ':' ')' '(' or two uppercase without comma:
                -->in latter case do not count term before last comma
        B.3     if no ', ' before first term: continue to the left until reaching lowercase, ':' ')' '(' ', '

        :param object:          string
        :return:                list of law firms, object without law firms
        '''

        list_of_lawfirms = []
        
        while 'Law Office of' in object:
            if ' ::: ' in object[
                          object.rfind('Law Office of'):object.rfind('Law Office of') + 20]: # already considered firm
                object      = ' ::: '.join(object.rsplit('Law Office of', 1))
            else:
                iter_beg    = object.rfind('Law Office of') + 14
                i           = object.rfind('Law Office of') + 15
                iter_end    = len(object)
                while i < len(object):  # iterate until comma
                    if object[i] == ',':
                        iter_end = i - 1
                        break
                    i += 1

                list_of_lawfirms += [object[iter_beg:iter_end + 1]]
                if object[iter_beg:iter_end + 1] != '':
                    object  = ' ::: '.join(object.rsplit(object[iter_beg:iter_end + 1], 1))
                object      = ' ::: '.join(object.rsplit('Law Office of', 1))        
        while 'Law Offices of' in object:
            if ' ::: ' in object[object.rfind('Law Offices of'):object.rfind(
                    'Law Offices of') + 21]:  # already considered law firm
                object = ' ::: '.join(object.rsplit('Law Offices of', 1))
            else:
                iter_beg    = object.rfind('Law Offices of') + 15
                i           = object.rfind('Law Offices of') + 16
                iter_end    = len(object)
                while i < len(object):  # iterate until comma
                    if object[i] == ',':
                        iter_end = i - 1
                        break
                    i   += 1

                list_of_lawfirms += [object[iter_beg:iter_end + 1]]
                if object[iter_beg:iter_end + 1] != '':
                    object  = ' ::: '.join(object.rsplit(object[iter_beg:iter_end + 1], 1))
                object      = ' ::: '.join(object.rsplit('Law Offices of', 1))
        
        lst = ['P.C.', 'PC', 'L.L.P.', 'LLP', 'P.L.L.C.', 'PLLC', 'P.L.C.', 'PLC', 'P.A.', 'Law Office', 'Law Offices']
        for str in lst:
            while object.rfind(str) != -1:
                iter_end    = object.rfind(str) - 1
                if ' ::: ' in object[object.rfind(str) - 7:object.rfind(str)]:
                    object  = ' ::: '.join(object.rsplit(str, 1))
                else:
                    if object[procede_until_lst_left(object, object.rfind(str) - 2, ' ') - 3:procede_until_lst_left(
                        object, object.rfind(str) - 2,' ')] == 'and':  # case with 'and'
                        object      = object[:procede_until_lst_left(object, object.rfind(str) - 2, ' ') - 3] + '&' + \
                        object[procede_until_lst_left(object,object.rfind(str) - 2,' '):]
                        iter_end    -= 3

                    if object[procede_until_lst_left(object, object.rfind(str) - 2, \
                                            ' ') - 1] != ',':  # case without comma before first term, associates
                        iter_obj    = procede_until_lst_left(object, object.rfind(str) - 3, \
                                                             [': ', '(', ') ', ', ', '; ']) + 1
                        iter_upper  = find_upper_after_lower(object, object.rfind(str) - 3)
                        iter_beg    = max(iter_obj, iter_upper)

                    else:  # case with comma before first term
                        iter_obj    = procede_until_lst_left(object, object.rfind(str) - 3, [': ', '(', ') ', '; ']) + 1
                        iter_upper  = find_upper_after_lower(object, object.rfind(str) - 3)
                        iter_2up    = two_upper_no_comma_left(object, object.rfind(str) - 3)
                        iter_beg    = max(iter_obj, iter_upper, iter_2up)

                    if iter_beg     == 1:
                        list_of_lawfirms += [object[iter_beg - 1:iter_end]]
                        object      = object.replace(object[iter_beg - 1:iter_end + 1], ' ::: ')
                    else:
                        list_of_lawfirms += [object[iter_beg:iter_end]]
                        object      = object.replace(object[iter_beg:iter_end + 1], ' ::: ')

                    object          = ' ::: '.join(object.rsplit(str, 1))
        return [list_of_lawfirms, object]
        

    # extraction of roles and law firms
    ####################################################################################################################    
    attorney_firms = []
    attorney_roles = []

    for j in range(len(attorney)):  ###iterate in attorney

        attorney[j] = re.sub(r'U\.S\.|U\. S\.', 'United States', attorney[j])   # cleaning U.S.--> United States
        attorney[j] = re.sub(r', Jr\.', ' Jr.', attorney[j])                    # cleaning ', Jr.'-->' Jr.'
        attorney[j] = re.sub(r'Atty\.', 'Attorney', attorney[j])                # cleaning Atty. --> Attorney
        attorney[j] = re.sub(r'Attys\.', 'Attorneys', attorney[j])

        attorneys_intermediate          = []            # roles in string
        firms_intermediate              = []            # firms in string

        if len(attorney) < 2:           # if len(attorney)<2 --> no data extraction
            attorneys_intermediate      += ['']
            firms_intermediate          += ['']
            break

        if ' for ' in attorney[j]:      # if ' for ' in string: extract following term and replace ' for '
            while ' for ' in attorney[j]:
                attorneys_intermediate  += [
                    term_expanded_general(attorney[j], attorney[j].rfind(' for ') + 7, [], [' ', '-'])]
                attorney[j]             = attorney[j].replace(' for ', ' ::: ')
        else:   attorneys_intermediate  += ['']

        if len(attorneys_intermediate)  > 1: attorneys_intermediate      = [''] # no unique role identifiable
        
        # extracting attorney general roles
        ################################################################################################################
        list_of_removals1           = ['Attorney General', 'Atty. Gen.', 'Attorney-General', 'Attys. Gen.',
                             'Attorneys General', 'Attorneys-General', 'Attorney- General',
                             'Attorney - General', 'Attorney -General', 'Attorney .General']  # (atty. gen. ;;;)
        for el in list_of_removals1:
            while attorney[j].find(el) != -1:
                firms_intermediate += ['Attorney General']
                attorney[j]         = attorney[j].replace(el, ' ::: ')
        ### extracting United States Attorney roles
        list_of_removals2           = ['United States Attorney', 'US Attorney', 'U.S. Attorney', 'U.S. Atty',
                             'U. S. Atty', 'U.S. Attorney', "State's Attorney", 'District Attorney',
                             'District Atty', "State's Atty", 'County Attorney', 'County Atty', "City Attorney",
                             'City Atty']
        for el in list_of_removals2:
            while attorney[j].find(el) != -1:
                firms_intermediate  += ['United States Attorney']
                attorney[j]         = attorney[j].replace(el, ' ::: ')
        ### extracting Public Defender roles
        while attorney[j].find('Public Defender') != -1:
            firms_intermediate      += ['Public Defender']
            attorney[j]             = attorney[j].replace('Public Defender', ' ::: ')

        ### extracting law firms containing a ' & '
        firms_intermediate          += find_replace_lawfirm_with(attorney[j])[0]
        attorney[j]                 = find_replace_lawfirm_with(attorney[j])[1]

        ### extracting law firms containing an abbreviation such as 'LLP'
        firms_intermediate          += find_replace_lawfirm_abbrev(attorney[j])[0]
        attorney[j]                 = find_replace_lawfirm_abbrev(attorney[j])[1]

        attorney_roles              += attorneys_intermediate
        attorney_firms              += [firms_intermediate]

    return attorney_roles, attorney_firms


def clustering_mechanism(all_strings, dis_threshold=2.7, matrix=None):
    '''
    Uses an agglomerative clustering mechanism to cluster strings based on their Levenshtein distance. The most frequent
    string of every cluster is assigned as a proxy for all strings in the cluster.

    :param all_strings:     pandas series containing the strings to be clustered
    :param dis_threshold:   threshold for the agglomerative clustering
    :param matrix:          matrix containing the distances between the strings
    :return:                list of strings where strings have been replaced by most frequent string of their cluster
    :return:                matrix containing the distances between the strings
    '''

    strings                 = pd.Index(all_strings.unique())              # unique strings
    
    # Compute matrix
    ####################################################################################################################
    if matrix is None:                                          # compute matrix
        n = len(strings)                                        # Number of strings
        logging.info(f'Started computation of matrix for {n} unique strings')
        matrix              = np.zeros((n, n))                  # Create an n x n matrix of zeros
        indices             = np.triu_indices(n, 1)             # Compute the upper triangular part of the matrix
        distances           = [Levenshtein.distance(strings[i],
                            strings[j], score_cutoff=1.4*dis_threshold) for i, j in zip(*indices)]

        matrix[indices]     = distances                         # Fill the matrix with the computed distances
        matrix              += matrix.T                         # Fill lower triangle part of the matrix
    
    # Perform Agglomerative Clustering
    ####################################################################################################################
    logging.info('Starting Agglomerative Clustering')

    clustering          = AgglomerativeClustering(n_clusters=None,metric='precomputed',distance_threshold=dis_threshold,
                                         linkage='average').fit(matrix)

    df                  = pd.DataFrame({'strings': all_strings})                                # initialise df
    df['label']         = df['strings'].map(lambda x: clustering.labels_[strings.get_loc(x)])   # add cluster labels
    
    # most frequent string of every cluster chosen as proxy for all strings
    ####################################################################################################################
    df['proxy_string']  = df.groupby('label')['strings'].transform(lambda x: x.value_counts().idxmax())

    return list(df['proxy_string']), matrix


def analyse_roles(cases, dis_threshold = 2.7, Mat=None):
    '''
    Extracts roles by formatting them and then calling clustering_mechanism to cluster them in order to account for
    spelling variations.

    :param cases:           list of cases
    :param dis_threshold:   threshold for the agglomerative clustering
    :param Mat:             matrix containing the distances between the strings
    :return:                cases with roles replaced by the most frequent string of their cluster
    :return:                matrix containing the distances between the strings
    '''
        
    # minor helper function
    ####################################################################################################################
    def plural_singular(object):  # turn plural to singular
        if object[len(object) - 1] != 's':  # no plural
            return object
        else:
            object = object[0:len(object) - 1]  # remove final 's'
            # object.replace('s-','-')
            while 's-' in object:  # remove intermediate plurals
                object = object.replace('s-', '-')
            while 'cro-' in object:  # consider case with 'cross' eg cross-appellant
                object = object.replace('cro-', 'cross')
            return object
            
    # extraction of roles
    ####################################################################################################################
    defi            = ['aintiff', 'efenda', 'ppell', 'omplain', 'etition', 'espond']
    list_roles      = [] # stores roles
    indices_roles   = [] # stores indices of attorney string and attorney substring

    for i in range(len(cases)):  # iterate over cases
        if i % 100000 == 0: logging.info(f'iterated all_roles until {i}')

        for j in range(len(cases[i][1])):  # iterate over expressions in each case
            k = 0                           # counts number of elements of defi not in a specific string
            for tx in defi:                 # itearte over all interesting strings via defi
                if tx in cases[i][1][j]:    # if match via defi
                    # match--> replace role of case by term representing it in singular; else--> nothing

                    ####################################################################################################
                    # BEGIN: extracting and slightly altering terms
                    cases[i][1][j]          = term_expanded_general(cases[i][1][j], cases[i][1][j].rfind(tx) + 2, [],
                                                           ['-']).lower()
                    # cases where role includes a ' ' not considered
                    cases[i][1][j]          = plural_singular(cases[i][1][j])
                    cases[i][1][j]          = cases[i][1][j].replace('-', '')   # remove '-'
                    if cases[i][1][j][0]    == ' ':                             # case where term starts with ' '
                        cases[i][1][j]      = cases[i][1][j][1:]
                    # END: extracting and slightly altering terms
                    ####################################################################################################
                else: k                     += 1            # tx not in cases[i][1][j] --> k+=1
            if k                            == len(defi):   # if syllable no match: put role to ''
                cases[i][1][j]              = ''
            
            list_roles      += [cases[i][1][j]]
            indices_roles   += [(i,j)]
    
    # clustering mechanism
    ####################################################################################################################
    list_most_frequent_string, matrix       = clustering_mechanism(\
        pd.Series(list_roles), dis_threshold=dis_threshold, matrix=Mat)
    for k in range(len(list_roles)):
        (i, j)          = indices_roles[k]              # choose indices of roles
        cases[i][1][j]  = list_most_frequent_string[k]  # replace attorney substring by list_most_frequent_string

    return cases, matrix


def analyse_firms(cases, dis_threshold = 2.7, Mat=None):
    '''
    Extracts law firms by formatting them and then calling clustering_mechanism to cluster them in order to account for
    spelling variations.
    
    :param cases:           list of cases
    :param dis_threshold:   threshold for the agglomerative clustering
    :param Mat:             matrix containing the distances between the strings
    :return:                cases with roles replaced by the most frequent string of their cluster
    :return:                matrix containing the distances between the strings
    '''
    
    # minor helper functions
    ####################################################################################################################
    def remove_hyphen(object):          # remove all '-' unless followed by capital
        i = 0
        while i < len(object) - 1:
            if object[i] == '-' and object[i + 1].isupper() != True:
                object = object[:i] + object[(i + 1):]
            i += 1
        return object

    def remove_sr_esq_firms(object):    # remove first 'sr.', then 'sr. ,', then ' esq.,'
        if object.find('sr., ')     == 0:       object = object[5:]
        if object.find('sr.,')      == 0:       object = object[4:]
        if object.find('sr.')       == 0:       object = object[3:]
        if object.find('sr. ')      == 0:       object = object[4:]
        if object.find('esq., ')    == 0:       object = object[6:]
        if object.find('esq. ')     == 0:       object = object[5:]
        return object

    def eliminate_firms(case): # Eliminates cases where unwanted symbols occur
        if len(case) != 0 and case[0] == '&':   case = ''
        ze = 0
        symb = ['&', ' ', ',', '.', '-', '’']
        for ze in range(len(case)):
            if case[ze].isalpha() == False and case[ze] not in symb:
                case = ''
                break
        return case
    
    # extraction of law firms
    ####################################################################################################################

    list_firms      = [] # stores roles
    indices_firms   = [] # stores indices of attorney string and attorney substring

    for i in  range(len(cases)):                    # iterate all cases
        if i % 100000 == 0: logging.info(f'iterated all_firms until {i}')

        for j in range(len(cases[i][2])):           # iterate expressions in each case
            for z in range(len(cases[i][2][j])):    # iterate potential law firms
                
                ########################################################################################################
                # BEGIN: altering terms
                if len(cases[i][2][j][z]) != 0 and cases[i][2][j][z][
                    -1] == ',':  # remove final ','   len(cases[i][2][j][z])
                    cases[i][2][j][z] = cases[i][2][j][z][:(len(cases[i][2][j][z]) - 1)]

                if len(cases[i][2][j][z]) != 0 and cases[i][2][j][z][0] == ' ':  # case where term starts with ' '
                    cases[i][2][j][z] = cases[i][2][j][z][1:]
                
                cases[i][2][j][z] = remove_hyphen(cases[i][2][j][z])
                # remove first 'sr.', then 'sr. ,', then ' esq.,', retain 'law group'
                cases[i][2][j][z] = cases[i][2][j][z].lower()  # everything to lowercase
                cases[i][2][j][z] = remove_sr_esq_firms(cases[i][2][j][z])      # remove 'sr.' and 'esq.'
                cases[i][2][j][z] = cases[i][2][j][z].replace('<&', '&')        # replace <& with &
                #eliminate law firms starting with '&' or with symbols
                cases[i][2][j][z] = eliminate_firms(cases[i][2][j][z])
                
                # remove legal abbreviations
                lst = [' p.c.', ' pc', ' l.l.p.', ' llp', ' p.l.l.c.', ' pllc', ' p.l.c.', ' plc', ' p.a.']
                for el in lst:
                    if cases[i][2][j][z][-len(el):] == el:
                        cases[i][2][j][z]=cases[i][2][j][z][:-len(el)]

                # remove name suffixes
                lst2=['jr,, ','jr, ','iii, ','ii, ','i, ','iv, ','v, ']
                for el in lst2:
                    if cases[i][2][j][z][:len(el)] == el:
                        cases[i][2][j][z] = cases[i][2][j][z][len(el):]
                # END: altering terms
                ########################################################################################################
                list_firms      += [cases[i][2][j][z]]
                indices_firms   += [(i, j, z)]
    
    # clustering mechanism
    ####################################################################################################################
    list_most_frequent_string, matrix       = clustering_mechanism(\
        pd.Series(list_firms), dis_threshold=dis_threshold, matrix=Mat)
    for k in range(len(list_firms)):
        (i, j, z)               = indices_firms[k]             # choose indices of roles
        cases[i][2][j][z]       = list_most_frequent_string[k] # replace attorney substring by list_most_frequent_string

    return cases, matrix


def add_extracted_columns_cases_df():
    '''
    Loads cases_df and adds two columns - extracted_roles and extracted_firms - to cases_df.
    This is done by extracting roles and firms from the attorney strings and then clustering them in order to
    account for spelling variations.
    The resulting cases_df is saved to the same location of the initially loaded cases_df.
    '''

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # read out the cases via get_path_cases_df(), extract firms and roles via extract_firms_roles(), further format and
    # cluster roles and firms via analyse_roles() and analyse_firms(), respectively
    ####################################################################################################################
    cases = []                      # cases are structured as [(ID, [roles,...], [[firms],...])]
    with gzip.open(f'{get_dir()}cases_df.csv.gz', mode='rt') as csv_file:           # open cases_df
        csvr    = csv.DictReader(csv_file, delimiter=',')

        for i, sdf in enumerate(csvr):
            if i % 10000 == 0: logging.info(f'Read cases until {i}.')               # logging progress
            attorney                        = ast.literal_eval(sdf['attorneys'])    # format attorney string

            attorney_roles, attorney_firms  = extract_firms_roles(attorney)         # extract law firms and roles
            cases                           += [(sdf['ID'], attorney_roles, attorney_firms)]    # append to cases

    # the roles and firms are further formatted and clustered separately
    cases,_ = analyse_roles(cases)  # format and cluster roles
    cases,_ = analyse_firms(cases)  # format and cluster law firms

    # add extracted_roles and extracted_firms to cases_df and then save it
    ####################################################################################################################
    fn                             = f'{get_dir()}cases_df.csv.gz'
    cases_df                       = pd.read_csv( fn , compression='gzip')
    df                             = pd.DataFrame(cases, columns=['ID', 'roles', 'firms'])
    cases_df['extracted_roles']    = df['roles']
    cases_df['extracted_firms']    = df['firms']
    _                              = cases_df.to_csv( fn, index=False, compression='gzip') # overwrite with extra info

if __name__=='__main__':

    add_extracted_columns_cases_df()