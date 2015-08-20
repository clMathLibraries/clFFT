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

import itertools
import re#gex
import subprocess
import os
import sys
from datetime import datetime

# Common data and functions for the performance suite

tableHeader = 'lengthx,lengthy,lengthz,batch,device,inlay,outlay,place,precision,label,GFLOPS'

class TestCombination:
    def __init__(self,
                 lengthx, lengthy, lengthz, batchsize,
                 device, inlayout, outlayout, placeness, precision,                 
                 label):
        self.x = lengthx
        self.y = lengthy
        self.z = lengthz
        self.batchsize = batchsize
        self.device = device
        self.inlayout = inlayout
        self.outlayout = outlayout
        self.placeness = placeness
        self.precision = precision
        self.label = label

    def __str__(self):
        return self.x + 'x' + self.y + 'x' + self.z + ':' + self.batchsize + ', ' + self.device + ', ' + self.inlayout + '/' + self.outlayout + ', ' + self.placeness + ', ' + self.precision + ' -- ' + self.label

class GraphPoint:
    def __init__(self,
                 lengthx, lengthy, lengthz, batchsize,
				 precision, device, label,
                 gflops):
        self.x = lengthx
        self.y = lengthy
        self.z = lengthz
        self.batchsize = batchsize
        self.device = device
        self.label = label
        self.precision = precision
        self.gflops = gflops
        self.problemsize = str(int(self.x) * int(self.y) * int(self.z) * int(self.batchsize))

    def __str__(self):
        # ALL members must be represented here (x, y, z, batch, device, label, etc)
        return self.x + 'x' + self.y + 'x' + self.z + ':' + self.batchsize + ', ' + self.precision + ' precision, ' + self.device + ', -- ' + self.label + '; ' + self.gflops

class TableRow:
    # parameters = class TestCombination instantiation
    def __init__(self, parameters, gflops):
        self.parameters = parameters
        self.gflops = gflops

    def __str__(self):
        return self.parameters.__str__() + '; ' + self.gflops

def transformDimension(x,y,z):
    if int(z) != 1:
        return 3
    elif int(y) != 1:
        return 2
    elif int(x) != 1:
        return 1

def executable(library):
    if type(library) != str:
        print 'ERROR: expected library name to be a string'
        quit()

    if sys.platform != 'win32' and sys.platform != 'linux2':
        print 'ERROR: unknown operating system'
        quit()

    if library == 'clFFT' or library == 'null':
        if sys.platform == 'win32':
            exe = 'clFFT-client.exe'
        elif sys.platform == 'linux2':
            exe = 'clFFT-client'
    elif library == 'cuFFT':
        if sys.platform == 'win32':
            exe = 'cuFFT-client.exe'
        elif sys.platform == 'linux2':
            exe = 'cuFFT-client'
    else:
        print 'ERROR: unknown library -- cannot determine executable name'
        quit()

    return exe

def max_mem_available_in_bytes(exe, device):
    arguments = [exe, '-i', device]
    
    deviceInfo = subprocess.check_output(arguments, stderr=subprocess.STDOUT).split(os.linesep)
    deviceInfo = itertools.ifilter( lambda x: x.count('MAX_MEM_ALLOC_SIZE'), deviceInfo)
    deviceInfo = list(itertools.islice(deviceInfo, None))
    maxMemoryAvailable = re.search('\d+', deviceInfo[0])
    return int(maxMemoryAvailable.group(0))

def max_problem_size(exe, layout, precision, device):

    if precision == 'single':
        bytes_in_one_number = 4
    elif precision == 'double':
        bytes_in_one_number = 8
    else:
        print 'max_problem_size(): unknown precision'
        quit()

    max_problem_size = pow(2,25)
    if layout == '5':
      max_problem_size = pow(2,24) # TODO: Upper size limit for real transform
    return max_problem_size

def maxBatchSize(lengthx, lengthy, lengthz, layout, precision, exe, device):
    problemSize = int(lengthx) * int(lengthy) * int(lengthz)
    maxBatchSize = max_problem_size(exe, layout, precision, device) / problemSize
    return str(maxBatchSize)

def create_ini_file_if_requested(args):
    if args.createIniFilename:
        for x in vars(args):
            if (type(getattr(args,x)) != file) and x.count('File') == 0:
                args.createIniFilename.write('--' + x + os.linesep)
                args.createIniFilename.write(str(getattr(args,x)) + os.linesep)
        quit()
    
def load_ini_file_if_requested(args, parser):
    if args.useIniFilename:
        argument_list = args.useIniFilename.readlines()
        argument_list = [x.strip() for x in argument_list]
        args = parser.parse_args(argument_list)
    return args

def is_numeric_type(x):
    return type(x) == int or type(x) == long or type(x) == float

def split_up_comma_delimited_lists(args):
    for x in vars(args):
        attr = getattr(args, x)
        if attr == None:
            setattr(args, x, [None])
        elif is_numeric_type(attr):
            setattr(args, x, [attr])
        elif type(attr) == str:
            setattr(args, x, attr.split(','))
    return args

class Range:
    def __init__(self, ranges, defaultStep='+1'):
        # we might be passed in a single value or a list of strings
        # if we receive a single value, we want to feed it right back
        if type(ranges) != list:
            self.expanded = ranges
        elif ranges[0] == None:
            self.expanded = [None]
        else:
            self.expanded = []
            for thisRange in ranges:
                thisRange = str(thisRange)
                if re.search('^\+\d+$', thisRange):
                    self.expanded = self.expanded + [thisRange]
                elif thisRange == 'max':
                    self.expanded = self.expanded + ['max']
                else:
                #elif thisRange != 'max':
                    if thisRange.count(':'):
                        self._stepAmount = thisRange.split(':')[1]
                    else:
                        self._stepAmount = defaultStep
                    thisRange = thisRange.split(':')[0]

                    if self._stepAmount.count('x'):
                        self._stepper = '_mult'
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

                    if self.begin == 0 and self._stepper == '_mult':
                        self.expanded = self.expanded + [0]
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

def expand_range(a_range):
    return Range(a_range).expanded

def decode_parameter_problemsize(problemsize):
    if not problemsize.count(None):
        i = 0
        while i < len(problemsize):
            problemsize[i] = problemsize[i].split(':')
            j = 0
            while j < len(problemsize[i]):
                problemsize[i][j] = problemsize[i][j].split('x')
                j = j+1
            i = i+1

    return problemsize

def gemm_table_header():
    return 'm,n,k,lda,ldb,ldc,alpha,beta,order,transa,transb,function,device,library,label,GFLOPS'

class GemmTestCombination:
    def __init__(self,
                 sizem, sizen, sizek, lda, ldb, ldc,
                 alpha, beta, order, transa, transb,
                 function, device, library, label):
        self.sizem = str(sizem)
        self.sizen = str(sizen)
        self.sizek = str(sizek)
        self.lda = str(lda)
        self.ldb = str(ldb)
        self.ldc = str(ldc)
        self.alpha = str(alpha)
        self.beta = str(beta)
        self.order = order
        self.transa = transa
        self.transb = transb
        self.function = function
        self.device = device
        self.library = library
        self.label = label

    def __str__(self):
        return self.sizem + 'x' + self.sizen + 'x' + self.sizek + ':' + self.lda + 'x' + self.ldb + 'x' + self.ldc + ', ' + self.device + ', ' + self.function + ', ' + self.library + ', alpha(' + self.alpha + '), beta(' + self.beta + '), order(' + self.order + '), transa(' + self.transa + '), transb(' + self.transb + ') -- ' + self.label

class GemmGraphPoint:
    def __init__(self,
                 sizem, sizen, sizek,
                 lda, ldb, ldc,
                 device, order, transa, transb,
                 function, library, label,
                 gflops):
        self.sizem = sizem
        self.sizen = sizen
        self.sizek = sizek
        self.lda = lda
        self.ldb = ldb
        self.ldc = ldc
        self.device = device
        self.order = order
        self.transa = transa
        self.transb = transb
        self.function = function
        self.library = library
        self.label = label
        self.gflops = gflops

    def __str__(self):
        # ALL members must be represented here (x, y, z, batch, device, label, etc)
        return self.sizem + 'x' + self.sizen + 'x' + self.sizek + ':' + self.device + ', ' + self.function + ', ' + self.library + ', order(' + self.order + '), transa(' + self.transa + '), transb(' + self.transb + ') -- ' + self.label + '; ' + self.gflops + ' gflops'

def open_file( filename ):
    if type(filename) == list:
        filename = filename[0]
    if filename == None:
        filename = 'results' + datetime.now().isoformat().replace(':','.') + '.txt'
    else:
        if os.path.isfile(filename):
            oldname = filename
            filename = filename + datetime.now().isoformat().replace(':','.')
            message = 'A file with the name ' + oldname + ' already exists. Changing filename to ' + filename
            print message
    
    return open(filename, 'w')
