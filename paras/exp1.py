#!/usr/bin/python

import sys


def main(argc, argv):
    print("argc = ", argc)
    print("argv = ", argv)


if __name__ == "__main__":
    main(len(sys.argv), str(sys.argv))

