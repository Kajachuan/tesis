import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mitad1", type=str, help="Directorio de la primera mitad")
    parser.add_argument("--mitad2", type=str, help="Directorio de la segunda mitad")
    parser.add_argument("--output", type=str, help="Directorio donde se guarda el resultado")
    args = parser.parse_args()

    f1 = open(args.mitad1)
    f2 = open(args.mitad2)

    data1 = json.load(f1)
    data2 = json.load(f2)

    f1.close()
    f2.close()

    for target1 in data1["targets"]:
        for target2 in data2["targets"]:
            if target2["name"] == target1["name"]:
                frames1 = target1["frames"]
                frames2 = target2["frames"]
                for frame in frames2:
                    frame["time"] += len(frames1)
                target1["frames"] = frames1 + frames2
                break

    with open(args.output, "w") as out:
        json.dump(data1, out, indent=2)

if __name__ == '__main__':
    main()
