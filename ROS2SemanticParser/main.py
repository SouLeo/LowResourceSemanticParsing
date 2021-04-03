from CopyNet import SemanticParser


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sr = SemanticParser()
    sr.predict_continuous("turn left")
    sr.predict_continuous("turn right")
    sr.predict_continuous("move forward")

