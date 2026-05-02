import json

def save_results(optimizer_name, accuracy):
    data = {
        "optimizer": optimizer_name,
        "accuracy": float(accuracy)
    }
    with open("results.json", "a") as f:
        f.write(json.dumps(data) + "\n")
