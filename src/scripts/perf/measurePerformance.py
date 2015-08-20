# ########################################################################
# Copyright 2013 Advanced Micro Devices, Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ########################################################################

import sys
import argparse
import subprocess
import itertools
import re#gex
import os
import math
from threading import Timer, Thread
import thread, time
from platform import system
import numpy as np

from datetime import datetime

import errorHandler
from fftPerformanceTesting import *
from performanceUtility import timeout, log, generate235Radices

IAM = 'FFT'
TIMOUT_VAL = 900  #In seconds
   
devicevalues = ['g', 'c']
layoutvalues = ['cp', 'ci']
placevalues = ['in', 'out']
precisionvalues = ['single', 'double']
libraryvalues = ['clFFT','cuFFT']
pow10 = '1-9,10-90:10,100-900:100,1000-9000:1000,10000-90000:10000,100000-900000:100000,1000000-9000000:1000000'

parser = argparse.ArgumentParser(description='Measure performance of the clFFT library')
parser.add_argument('--device',
    dest='device', default='g',
    help='device(s) to run on; may be a comma-delimited list. choices are ' + str(devicevalues) + '. (default gpu)')
parser.add_argument('-b', '--batchsize',
    dest='batchSize', default='1',
    help='number of FFTs to perform with one invocation of the client. the special value \'adapt\' may be used to adjust the batch size on a per-transform basis to the maximum problem size possible on the device. (default 1)'.format(pow10))
parser.add_argument('-a', '--adaptivemax',
    dest='constProbSize', default='-1',
    help='Max problem size that you want to maintain across the invocations of client with different lengths. This is adaptive and adjusts itself automtically.'.format(pow10))
parser.add_argument('-x', '--lengthx',
    dest='lengthx', default='1',
    help='length(s) of x to test; must be factors of 1, 2, 3, or 5 with clFFT; may be a range or a comma-delimited list. e.g., 16-128 or 1200 or 16,2048-32768 (default 1)')
parser.add_argument('-y', '--lengthy',
    dest='lengthy', default='1',
    help='length(s) of y to test; must be factors of 1, 2, 3, or 5 with clFFT; may be a range or a comma-delimited list. e.g., 16-128 or 1200 or 16,32768 (default 1)')
parser.add_argument('-z', '--lengthz',
    dest='lengthz', default='1',
    help='length(s) of z to test; must be factors of 1, 2, 3, or 5 with clFFT; may be a range or a comma-delimited list. e.g., 16-128 or 1200 or 16,32768 (default 1)')
parser.add_argument('-reps',
    dest='reps', default='10',
    help='Number of repetitions (default 10)')
parser.add_argument('-prime_factor', '--prime_factor',
    dest='prime_factor', default='2',
    help='only test the prime factors within the specified range of lengthx/y/z. Select from 2,3,5, and 7. Example: -prime_factor 2,3')
parser.add_argument('-test_count', '--test_count',
    dest='test_count', default='100',
    help='Number of tests to perform')
parser.add_argument('--problemsize',
    dest='problemsize', default=None)
#    help='additional problems of a set size. may be used in addition to lengthx/y/z. each indicated problem size will be added to the list of FFTs to perform. should be entered in AxBxC:D format. A, B, and C indicate the sizes of the X, Y, and Z dimensions (respectively). D is the batch size. All values except the length of X are optional. may enter multiple in a comma-delimited list. e.g., 2x2x2:32768 or 256x256:100,512x512:256')
parser.add_argument('-i', '--inputlayout',
    dest='inputlayout', default='1',
    help=' 1. interleaved (default) 2. planar  3. hermitian interleaved 4. hermitian planar  5. real' )
parser.add_argument('-o', '--outputlayout',
    dest='outputlayout', default='1',
    help=' 1. interleaved (default) 2. planar  3. hermitian interleaved 4. hermitian planar  5. real' )
parser.add_argument('--placeness',
    dest='placeness', default='in',
    help='Choices are ' + str(placevalues) + '. in = in place, out = out of place (default in)')
parser.add_argument('-r', '--precision',
    dest='precision', default='single',
    help='Choices are ' + str(precisionvalues) + '. (default single)')
parser.add_argument('--library',
    dest='library', default='clFFT', choices=libraryvalues,
    help='indicates the library to use for testing on this run')
parser.add_argument('--label',
    dest='label', default=None,
    help='a label to be associated with all transforms performed in this run. if LABEL includes any spaces, it must be in \"double quotes\". note that the label is not saved to an .ini file. e.g., --label cayman may indicate that a test was performed on a cayman card or --label \"Windows 32\" may indicate that the test was performed on Windows 32')
#parser.add_argument('--createini',
#    dest='createIniFilename', default=None,
#    help='create an .ini file with the given name that saves the other parameters given at the command line, then quit. e.g., \'measureperformance.py -x 2048 --createini my_favorite_setup.ini\' will create an .ini file that will save the configuration for a 2048-datapoint 1D FFT.')
parser.add_argument('--ini',
    dest='iniFilename', default=None,
    help='use the parameters in the named .ini file instead of the command line parameters.')
parser.add_argument('--tablefile',
    dest='tableOutputFilename', default=None,
    help='save the results to a plaintext table with the file name indicated. this can be used with plotPerformance.py to generate graphs of the data (default: table prints to screen)')
parser.add_argument('--prefix',
    dest='prefix', default='./',
    help='Path where the library client is located (default current directory)')

args = parser.parse_args()

label = str(args.label)

subprocess.call('mkdir perfLog', shell = True)
logfile = os.path.join('perfLog', (label+'-'+'fftMeasurePerfLog.txt'))

def printLog(txt):
    print txt
    log(logfile, txt)

printLog("=========================MEASURE PERFORMANCE START===========================")
printLog("Process id of Measure Performance:"+str(os.getpid()))

currCommandProcess = None


printLog('Executing measure performance for label: '+str(label))


#This function is defunct now
@timeout(1, "fileName") # timeout is 5 minutes, 5*60 = 300 secs
def checkTimeOutPut2(args):
    global currCommandProcess
    #ret = subprocess.check_output(args, stderr=subprocess.STDOUT)
    #return ret
    currCommandProcess = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    printLog("Curr Command Process id = "+str(currCommandProcess.pid))
    ret = currCommandProcess.communicate()    
    if(ret[0] == None or ret[0] == ''):
        errCode = currCommandProcess.poll()
        raise subprocess.CalledProcessError(errCode, args, output=ret[1])
    return ret[0]


#Spawns a separate thread to execute the library command and wait for that thread to complete
#This wait is of 900 seconds (15 minutes). If still the thread is alive then we kill the thread
def checkTimeOutPut(args):
    t = None
    global currCommandProcess
    global stde
    global stdo
    stde = None
    stdo = None
    def executeCommand():
        global currCommandProcess
        global stdo
        global stde
        try:
            stdo, stde = currCommandProcess.communicate()
            printLog('stdout:\n'+str(stdo))
            printLog('stderr:\n'+str(stde))
        except:
            printLog("ERROR: UNKNOWN Exception - +checkWinTimeOutPut()::executeCommand()")

    currCommandProcess = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
    thread = Thread(target=executeCommand)
    thread.start()
    thread.join(TIMOUT_VAL) #wait for the thread to complete 
    if thread.is_alive():
        printLog('ERROR: Killing the process - terminating thread because it is taking too much of time to execute')
        currCommandProcess.kill()
        printLog('ERROR: Timed out exception')
        raise errorHandler.ApplicationException(__file__, errorHandler.TIME_OUT)
    if stdo == "" or stdo==None:
        errCode = currCommandProcess.poll()
        printLog('ERROR: @@@@@Raising Called processor exception')
        raise subprocess.CalledProcessError(errCode, args, output=stde)
    return stdo





# don't try to create and use an .ini file at the same time (it will open a portal through which demons will emerge)
#if args.iniFilename and args.createIniFilename:
#    printLog('ERROR: --ini and --createini are mutually exclusive. Please choose only one.')
#    quit()

#read in .ini parameters if --ini is used
#if args.iniFilename != None:
#    if not os.path.isfile(args.iniFilename):
#        printLog("No file with the name \'{}\' exists. Please indicate another filename.".format(args.iniFilename))
#        quit()
#    
#    ini = open(args.iniFilename, 'r')
#    iniContents = ini.read()
#    iniContents = iniContents.split(';')
#    for i in range(0,len(iniContents)):
#        line = iniContents.pop()
#        line = line.partition(' ')
#        parameter = line[0]
#        value = line[2]
#        value = value.replace('\'','').replace('[','').replace(']','').replace(' ','')
#        print"value= ",value
#        
#        if parameter == 'batchSize':
#            args.batchSize = value
#        elif parameter == 'constProbSize':
#            args.constProbSize = value
#        elif parameter == 'lengthx':
#            args.lengthx = value
#        elif parameter == 'reps':
#            args.reps = value
#        elif parameter == 'prime_factor':
#            args.prime_factor = value
#        elif parameter == 'test_count':
#            args.test_count = value
#        elif parameter == 'lengthy':
#            args.lengthy = value
#        elif parameter == 'lengthz':
#            args.lengthz = value
#        elif parameter == 'problemsize':
#            args.problemsize = value
#        elif parameter == 'device':
#            args.device = value
#        elif parameter == 'inputlayout':
#            args.inputlayout = value
#        elif parameter == 'outputlayout':
#            args.outputlayout = value
#        elif parameter == 'placeness':
#            args.placeness = value
#        elif parameter == 'precision':
#            args.precision = value
#        else:
#            printLog('{} corrupted. Please re-create a .ini file with the --createini flag.'.format(args.iniFilename))
#            #quit()

#create ini file if requested
#if args.createIniFilename != None:
#    printLog('Creating Ini files')
#    if os.path.isfile(args.createIniFilename):
#        printLog('A file with the name \'{}\' already exists. Please delete the file or choose another name.'.format(args.createIniFilename))
#        quit()
#    printLog('Creating Ini file:'+args.createIniFilename+'\n')
#    ini = open(args.createIniFilename, 'w')
#    ini.write('batchSize {} ;'.format(args.batchSize))
#    ini.write('constProbSize {} ;'.format(args.constProbSize))
#    ini.write('lengthx {} ;'.format(args.lengthx))
#    ini.write('lengthy {} ;'.format(args.lengthy))
#    ini.write('lengthz {} ;'.format(args.lengthz))
#    ini.write('prime_factor {} ;'.format(args.prime_factor))
#    ini.write('test_count {} ;'.format(args.test_count))
#    ini.write('reps {} ;'.format(args.reps))
#    ini.write('problemsize {} ;'.format(args.problemsize))
#    ini.write('device {} ;'.format(args.device))
#    ini.write('inputlayout {} ;'.format(args.inputlayout))
#    ini.write('outputlayout {} ;'.format(args.outputlayout))
#    ini.write('placeness {} ;'.format(args.placeness))
#    ini.write('precision {} ;'.format(args.precision))
#    printLog('Created Ini file:'+args.createIniFilename+'\n')
#    printLog("=========================MEASURE PERFORMANCE START===========================\n")
#    quit()
#

#turn pow10 into its range list
if args.batchSize.count('pow10'):
    args.batchSize = pow10

#split up comma-delimited lists
args.batchSize = args.batchSize.split(',')
args.constProbSize = int(args.constProbSize.split(',')[0])
args.device = args.device.split(',')
args.lengthx = args.lengthx.split(',')
args.lengthy = args.lengthy.split(',')
args.lengthz = args.lengthz.split(',')
args.prime_factor = args.prime_factor.split(',')
if args.problemsize:
    args.problemsize = args.problemsize.split(',')
args.inputlayout = args.inputlayout.split(',')
args.outputlayout = args.outputlayout.split(',')
args.placeness = args.placeness.split(',')
args.precision = args.precision.split(',')



printLog('Executing for label: '+str(args.label))
#check parameters for sanity

# batchSize of 'max' must not be in a list (does not get on well with others)
#if args.batchSize.count('max') and len(args.batchSize) > 1:
if ( args.batchSize.count('max') or args.batchSize.count('adapt') )and len(args.batchSize) > 1:
    printLog('ERROR: --batchsize max must not be in a comma delimited list')
    quit()


# in case of an in-place transform, input and output layouts must be the same (otherwise: *boom*)
#for n in args.placeness:
#    if n == 'in' or n == 'inplace':
#        if len(args.inputlayout) > 1 or len(args.outputlayout) > 1 or args.inputlayout[0] != args.outputlayout[0]:
#            printLog('ERROR: if transformation is in-place, input and output layouts must match')
#            quit()

# check for valid values in precision
for n in args.precision:
    if n != 'single' and n != 'double':
        printLog('ERROR: invalid value for precision')
        quit()

def isPrime(n):
    import math
    n = abs(n)
    i = 2
    while i <= math.sqrt(n):
        if n%i == 0:
            return False
        i += 1
    return True

def findFactors(number):
    iter_space = range(1, number+1)
    prime_factor_list = []
    for curr_iter in iter_space:
        if isPrime(curr_iter) == True:
            #print 'curr_iter_prime: ', curr_iter
            if number%curr_iter == 0:
                prime_factor_list.append(curr_iter)
    return prime_factor_list


#Type : Function
#Input: num, a number which we need to factorize
#Return Type: list
#Details: This function returns only the prime factors on an input number
#         e.g: input: 20, returns: [2,2,5]
#              input: 32, returns: [2,2,2,2,2]
def factor(num):
    if num == 1:
        return [1]
    i = 2
    limit = num**0.5
    while i <= limit:
        if num % i == 0:
            ret = factor(num/i)
            ret.append(i)
            return ret
        i += 1
    return [num]

def validateFactors(flist):
    ref_list = [1,2,3,5]
    if flist==ref_list:
        return True
    if len(flist) > len(ref_list):
        return False
    for felement in flist:
        if ref_list.count(felement) != 1:
            return False
    return True

#Type : Function
#Input: num, a number which we need to validate for 1,2,3 or 5 factors
#Return Type: boolean
#Details: This function validates an input number for its prime factors
#         If factors has number other than 1,2,3 or 5 then return false else return true
#         e.g: input: 20, returns: True
#              input: 28, returns: False
def validate_number_for_1235(num):
    if num == 0:
        return True
    set1235 = set([1,2,3,5])
    setPrimeFactors = set(factor(num))
    setPrimeFactors = setPrimeFactors | set1235 #performed union of two sets
    #if still the sets are same then we are done!!!
    #else we got few factors other than 1,2,3 or 5 and we should invalidate
    #the input number
    if setPrimeFactors ==  set1235:
        return True
    return False


def getValidNumbersInRange(rlist):
    valid_number_list = []
    for relement in rlist:
        prime_factors = findFactors(relement)
        if validateFactors(prime_factors) == True:
            valid_number_list.append(relement)
    return valid_number_list

def get_next_num_with_1235_factors(start):
    start+=1
    while not validateFactors(findFactors(start)):
        start+=1
    return start


def check_number_for_1235_factors(number):
    #printLog('number:'+ number)
    factors = findFactors(number)
    #printLog('factors:'+ factors)
    if not validateFactors(factors):
        printLog("ERROR: --{0} must have only 1,2,3,5 as factors")
        return False
    return True



def check_for_1235_factors(values, option):
    #print 'values: ', values
    for n in values:
        for m in n.replace('-',',').split(','):
            if not validate_number_for_1235(int(m)):
                print 'ERROR: --{0} must specify number with only 1,2,3,5 as factors'.format(option)
                quit()
            #print 'Valid number for :',option,':', m
       

if args.library == 'clFFT':
    check_for_1235_factors(args.lengthx, 'lengthx')
    check_for_1235_factors(args.lengthy, 'lengthy')
    check_for_1235_factors(args.lengthz, 'lengthz')



if not os.path.isfile(args.prefix+executable(args.library)):
    printLog("ERROR: Could not find client named {0}".format(executable(args.library)))
    quit()


def get235RadicesNumberInRange(minimum, maximum):
    if minimum == 0 and maximum == 0:
        return [0]
    numbers = generate235Radices(maximum)
    minIndex = numbers.index(minimum)
    maxIndex = numbers.index(maximum)
    return numbers[minIndex:maxIndex+1]
   
#expand ranges
class Range:
    def __init__(self, ranges, defaultStep='+1'):
        self.expanded = []
        for thisRange in ranges:
            if thisRange != 'max' and thisRange != 'adapt' :
                if thisRange.count(':'):
                    self._stepAmount = thisRange.split(':')[1]
                else:
                    self._stepAmount = defaultStep
                thisRange = thisRange.split(':')[0]

                if self._stepAmount.count('x'):
                    self._stepper = '_mult'
                    self._stepAmount = self._stepAmount.lstrip('+x')
                    self._stepAmount = int(self._stepAmount)
                elif self._stepAmount.count('l'):
                    self._stepper = '_next_num_with_1235_factor'
                    self._stepAmount = 0
                else:
                    self._stepper = '_add'
                    self._stepAmount = self._stepAmount.lstrip('+x')
                    self._stepAmount = int(self._stepAmount)

                if thisRange.count('-'):
                    self.begin = int(thisRange.split('-')[0])
                    self.end = int(thisRange.split('-')[1])
                else:
                    self.begin = int(thisRange.split('-')[0])
                    self.end = int(thisRange.split('-')[0])
                self.current = self.begin

           # _thisRangeExpanded = []
            if thisRange == 'max':
                self.expanded = self.expanded + ['max']
            elif thisRange == 'adapt':
                self.expanded = self.expanded + ['adapt']
            elif self.begin == 0 and self._stepper == '_mult':
                self.expanded = self.expanded + [0]
            else:
                if self._stepper == '_next_num_with_1235_factor':
                    self.expanded = self.expanded + get235RadicesNumberInRange(self.current, self.end)
                else:
                    while self.current <= self.end:
                        self.expanded = self.expanded + [self.current]
                        self._step()

            # now we want to uniquify and sort the expanded range
            self.expanded = list(set(self.expanded))
            self.expanded.sort()

    # advance current value to next
    def _step(self):
        getattr(self, self._stepper)()

    def _mult(self):
        self.current = self.current * self._stepAmount

    def _add(self):
        self.current = self.current + self._stepAmount

    def _next_num_with_1235_factor(self):
        self.current = get_next_num_with_1235_factors(self.current)


args.batchSize = Range(args.batchSize).expanded
args.lengthx = Range(args.lengthx, 'l').expanded
args.lengthy = Range(args.lengthy, 'l').expanded
args.lengthz = Range(args.lengthz, 'l').expanded


def create_prime_factors(args,input_list):
  powers2=[1]
  powers3=[1]
  powers5=[1]
  powers7=[1]
  if '2' in args.prime_factor:
    powers2+=[2**x for x in range(1,int(math.floor(math.log(max(input_list),2)+1)))]
  if '3' in args.prime_factor:
    powers3+=[3**x for x in range(1,int(math.floor(math.log(max(input_list),3)+1)))]
  if '5' in args.prime_factor:
    powers5+=[5**x for x in range(1,int(math.floor(math.log(max(input_list),5)+1)))]
  if '7' in args.prime_factor:
    powers7+=[7**x for x in range(1,int(math.floor(math.log(max(input_list),7)+1)))]
  
  
  xlist=[]
  for i in powers2:
    for j in powers3:
      for k in powers5:
        for l in powers7:
          dummy=int(i)*int(j)*int(k)*int(l)
          if(dummy<=max(input_list)) and (dummy>=min(input_list)):
            xlist.append(dummy)
          
  xlist=sorted(xlist)
  xlist=xlist[:int(args.test_count)] #snafu
  return xlist

args.lengthx=create_prime_factors(args,args.lengthx)
args.lengthy=create_prime_factors(args,args.lengthy)
args.lengthz=create_prime_factors(args,args.lengthz)

#expand problemsizes ('XxYxZ:batch')
#print "args.problemsize--1-->", args.problemsize
if args.problemsize and args.problemsize[0] != 'None':
    i = 0
    while i < len(args.problemsize):
        args.problemsize[i] = args.problemsize[i].split(':')
        args.problemsize[i][0] = args.problemsize[i][0].split('x')
        i = i+1


#create the problem size combinations for each run of the client
# A: This part creats a product of all possible combinations. Too many cases in 2/3D
#problem_size_combinations = itertools.product(args.lengthx, args.lengthy, args.lengthz, args.batchSize)
#problem_size_combinations = list(itertools.islice(problem_size_combinations, None))

if args.lengthy[0]==1:
  args.lengthy=[1]*len(args.lengthx)
if args.lengthz[0]==1:
  args.lengthz=[1]*len(args.lengthx)

dummy=[args.batchSize[0]]*len(args.lengthx)
problem_size_combinations=zip(args.lengthx,args.lengthy,args.lengthz,dummy)

#print "args.problemsize--2-->", args.problemsize
#add manually entered problem sizes to the list of FFTs to crank out
manual_test_combinations = []
if args.problemsize and args.problemsize[0] != 'None':
    for n in args.problemsize:
        x = []
        y = []
        z = []
        batch = []
    
        x.append(int(n[0][0]))
    
        if len(n[0]) >= 2:
            y.append(int(n[0][1]))
        else:
            y.append(1)
    
        if len(n[0]) >= 3:
            z.append(int(n[0][2]))
        else:
            z.append(1)
    
        if len(n) > 1:
            batch.append(int(n[1]))
        else:
            batch.append(1)
    
        combos = itertools.product(x, y, z, batch)
        combos = list(itertools.islice(combos, None))
        for n in combos:
            manual_test_combinations.append(n)
        # manually entered problem sizes should not be plotted (for now). they may still be output in a table if requested


problem_size_combinations = problem_size_combinations + manual_test_combinations

#create final list of all transformations (with problem sizes and transform properties)
test_combinations = itertools.product(problem_size_combinations, args.device, args.inputlayout, args.outputlayout, args.placeness, args.precision)
test_combinations = list(itertools.islice(test_combinations, None))
test_combinations = [TestCombination(params[0][0], params[0][1], params[0][2], params[0][3], params[1], params[2], params[3], params[4], params[5], args.label) for params in test_combinations]

if args.iniFilename != None:
  array=np.genfromtxt(args.iniFilename, names=True, delimiter=',', dtype=None)
  test_combinations = [TestCombination(params[0],params[1], params[2], params[3], params[4],params[5],params[6],params[7],params[8],args.label) for params in array]

#print("lenghtx= ",test_combinations[0].x)
#print("lenghty= ",test_combinations[0].y)
#print("lenghtz= ",test_combinations[0].z)
#print("placeness= ",test_combinations[0].placeness)



#turn each test combination into a command, run the command, and then stash the gflops
result = [] # this is where we'll store the results for the table


#open output file and write the header

if args.tableOutputFilename == None:
  if args.library == 'cuFFT':
    args.tableOutputFilename = 'cuFFT_' + 'x_'+ str(args.lengthx[0]) + '_y_'+str(args.lengthy[0])+'_z_'+str(args.lengthz[0])+'_'+str(args.precision[0]) +'_'+datetime.now().isoformat().replace(':','.') + '.txt'
  elif args.library=='clFFT':
    args.tableOutputFilename = 'clFFT_' + 'x_'+ str(args.lengthx[0]) + '_y_'+str(args.lengthy[0])+'_z_'+str(args.lengthz[0])+'_'+str(args.precision[0])+ '_'+datetime.now().isoformat().replace(':','.') + '.txt'
else:
   if os.path.isfile(args.tableOutputFilename):
       oldname = args.tableOutputFilename
       args.tableOutputFilename = args.tableOutputFilename + datetime.now().isoformat().replace(':','.')
       message = 'A file with the name ' + oldname + ' already exists. Changing filename to ' + args.tableOutputFilename
       printLog(message)


printLog('table header---->'+ str(tableHeader))

table = open(args.tableOutputFilename, 'w')
table.write(tableHeader + '\n')
table.flush()
if args.constProbSize == -1:
   args.constProbSize = maxBatchSize(1, 1, 1, args.inputlayout[0], args.precision[0], executable(args.library), '-' + args.device[0])
args.constProbSize = int(args.constProbSize)


printLog('Total combinations =  '+str(len(test_combinations)))

vi = 0
for params in test_combinations:
    if vi>=int(args.test_count):
      break
    vi = vi+1
    printLog("")
    printLog('preparing command: '+ str(vi))    
    device = params.device
    lengthx = str(params.x)
    lengthy = str(params.y)
    lengthz = str(params.z)
    inlayout=str(params.inlayout)
    outlayout=str(params.outlayout)
    prefix=str(args.prefix)

    
    if params.batchsize == 'max':
        batchSize = maxBatchSize(lengthx, lengthy, lengthz, params.inlayout, params.precision, executable(args.library), '-' + device)
    elif params.batchsize == 'adapt':
        batchSize = str(args.constProbSize/(int(lengthx)*int(lengthy)*int(lengthz)))
    else:
        batchSize = str(params.batchsize)

    if params.placeness == 'inplace' or params.placeness == 'in':
        placeness = ''
    elif params.placeness == 'outofplace' or params.placeness == 'out':
        placeness = '--outPlace'
    else:
        printLog('ERROR: invalid value for placeness when assembling client command')

    if params.precision == 'single':
        precision = ''
    elif params.precision == 'double':
        precision = '--double'
    else:
        printLog('ERROR: invalid value for precision when assembling client command')


    #set up arguments here
    if args.library == 'clFFT':
        arguments = [prefix+ executable(args.library),
                     '-' + device,
                     '-x', lengthx,
                     '-y', lengthy,
                     '-z', lengthz,
                     '--batchSize', batchSize,
                     '--inLayout', inlayout,
                     '--outLayout',outlayout,
                     placeness,
                     precision,
                     '-p', args.reps]
    elif args.library == 'cuFFT':
        if  inlayout[0]=='1'  and outlayout[0]=='1':
          cuFFT_type='1'
        elif inlayout[0]=='3' and outlayout[0]=='5':
          cuFFT_type='3'
        elif inlayout[0]=='5' and outlayout[0]=='3':
          cuFFT_type='2'
        else:
          print"Wrong input/outputlayout. Only C2C/R2C/C2R are supported for Cuda"
          exit()
        arguments=[prefix+executable(args.library),
                     '-x', lengthx,
                     '-y', lengthy,
                     '-z', lengthz,
                     '-b', batchSize,
                     '-p',args.reps,
                     '-d',str(int(args.precision[0]=='double')),
                     '-type',cuFFT_type]
   
    writeline = True
    try:
        arguments=' '.join(arguments)
        printLog('Executing Command: '+str(arguments))
        output = checkTimeOutPut(arguments)
        output = output.split(os.linesep);
        printLog('Execution Successfull---------------\n')

    except errorHandler.ApplicationException as ae:
        writeline = False
        printLog('ERROR: Command is taking too much of time '+ae.message+'\n'+'Command: \n'+str(arguments))
        continue
    except subprocess.CalledProcessError as clientCrash:
        print 'Command execution failure--->'
        if clientCrash.output.count('CLFFT_INVALID_BUFFER_SIZE'):
            writeline = False
            printLog('Omitting line from table - problem is too large')
        else:
            writeline = False
            printLog('ERROR: client crash. Please report the following error message (with \'CLFFT_*\' error code, if given, and the parameters used to invoke measurePerformance.py) \n'+clientCrash.output+'\n')
            printLog('IN ORIGINAL WE CALL QUIT HERE - 1\n')
            continue

    for x in output:
        if x.count('out of memory'):
            writeline = False
            printLog('ERROR: Omitting line from table - problem is too large')

    if writeline:
        try:
            if args.library == 'cuFFT':
              output = itertools.ifilter( lambda x: x.count('gflops'), output)
            else:
              output = itertools.ifilter( lambda x: x.count('gflops'), output)

            output = list(itertools.islice(output, None))
            thisResult = re.search('\d+\.*\d*e*-*\d*$', output[-1])
            if args.library == 'cuFFT':
                thisResult = re.search('[-+]?\d*\.\d+|\d+$', output[-1])
            thisResult = float(thisResult.group(0))

            thisResult = (params.x, params.y, params.z, batchSize, params.device, params.inlayout, params.outlayout, params.placeness, params.precision, params.label, thisResult)

            outputRow = ''
            for x in thisResult:
                outputRow = outputRow + str(x) + ','
            outputRow = outputRow.rstrip(',')
            table.write(outputRow + '\n')
            table.flush()
        except:
			printLog('ERROR: Exception occurs in GFLOP parsing')
    else:
        if(len(output) > 0):
            if output[0].find('nan') or output[0].find('inf'):
                printLog( 'WARNING: output from client was funky for this run. skipping table row')
            else:
                prinLog('ERROR: output from client makes no sense')
                printLog(str(output[0]))
                printLog('IN ORIGINAL WE CALL QUIT HERE - 2\n')
        else:
            prinLog('ERROR: output from client makes no sense')
            #quit()
printLog("=========================MEASURE PERFORMANCE ENDS===========================\n")
#
#"""
#print a pretty table
#"""
#if args.tableOutputFilename == None:
#   args.tableOutputFilename = 'results' + datetime.now().isoformat().replace(':','.') + '.txt'
#else:
#   if os.path.isfile(args.tableOutputFilename):
#       oldname = args.tableOutputFilename
#       args.tableOutputFilename = args.tableOutputFilename + datetime.now().isoformat().replace(':','.')
#       message = 'A file with the name ' + oldname + ' already exists. Changing filename to ' + args.tableOutputFilename
#       print message
#
#table = open(args.tableOutputFilename, 'w')
#table.write(tableHeader + '\n')
#for x in result:
#   row = ''
#   for y in x:
#       row = row + str(y) + ','
#   row = row[:-1] #chomp off the trailing comma
#   table.write(row + '\n')
