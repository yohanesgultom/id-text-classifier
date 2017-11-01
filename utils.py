import os
import sys
import signal
import itertools

def print_titles(file, inc=3):
  """
  Print titles in a file
  
  Keyword arguments:
  real -- the real part (default 0.0)
  imag -- the imaginary part (default 0.0)
  """
  count = 0
  with open(file) as f:
      for line in f:
        if count == 0 or count % 3 == 0:
          print(line.strip() + '\t 0')
        count += 1


def unique_everseen(iterable, key=None):
    """
    List unique elements, preserving order. Remember all elements ever seen.
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    """
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in itertools.ifilterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def remove_duplicates(file):
  """
  Remove duplicate lines in file
  """
  file_tmp = 'tmp'
  with open(file) as f, open(file_tmp, 'w') as o:
    for line in unique_everseen(f):
      o.write(line)
  # rename file_tmp to file
  os.remove(file)
  os.rename(file_tmp, file)


def annotate_titles(file, start_line=0, default_annotation=0):
  """
  Print lines one by one and prompt annotation
  """
  count = 0
  with open(file) as f:
    for line in f:
      if count >= start_line:
        text, annotation = line.split('\t')
        print(text)
        new_annotation = raw_input('Annotation: ')
        if not new_annotation:
          break
      if count >= 10:
        break
      count += 1



if __name__ == '__main__':
  # # print titles from dataset
  # file = sys.argv[1]
  # print_titles(file)

  # # annotate lines
  # file = sys.argv[1]
  # start_line = 0
  # if len(sys.argv) > 2:
  #   start_line = int(sys.argv[2])
  # annotate_titles(file, start_line)  
  # # signal.signal(signal.SIGINT, save_file)
  # # signal.signal(signal.SIGTERM, save_file)

  remove_duplicates(sys.argv[1])

    