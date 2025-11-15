import modal, os

app = modal.App()

volume = modal.Volume.from_name("chess-data")

@app.function(volumes={"/data": volume})
def ls():
    return os.listdir("/data/train")

if __name__ == "__main__":
    print(ls.local())