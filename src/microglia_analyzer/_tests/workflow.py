from microglia_analyzer.microglia_analyzer import MicrogliaAnalyzer

def workflow_classic():
    def logging(message):
        print("[MA] " + message)
    target = "/home/benedetti/Documents/projects/2060-microglia/adulte 4.tif"
    mam = MicrogliaAnalyzer(logging)
    mam.load_image(target)


if __name__ == "__main__":
    workflow_classic()