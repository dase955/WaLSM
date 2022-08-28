import sys


def get_statistics(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    trivial_move = [0 for i in range(7)]
    flush_and_compaction = [0 for i in range(7)]
    flush_base = 0

    for line in lines:
        data = line.split()
        if line.startswith("Compaction"):
            level = int(data[-1])
            t = float(data[5][:-2])
            flush_and_compaction[level] += float(data[1][:-3]) / 1024.0
        elif line.startswith("Flush l0"):
            t = float(data[4][:-2])
            flush_and_compaction[0] += float(data[2][:-3]) / 1024.0
        elif line.startswith("Trivial move"):
            level = int(data[-1])
            t = float(data[3][:-2])
            trivial_move[level] += float(data[2][:-3]) / 1024.0
        elif line.startswith("Flush to base"):
            t = float(data[5][:-2])
            flush_base += float(data[3][:-3]) / 1024.0

    print("Flush to base: {:.2f}G".format(flush_base))
    for l in range(7):
        print("Level {} Trivial move: {:>6.2f}G, compaction: {:>6.2f}G".
              format(l, trivial_move[l], flush_and_compaction[l]))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Input file missing")
        sys.exit()

    file_name = sys.argv[1]
    get_statistics(file_name)
