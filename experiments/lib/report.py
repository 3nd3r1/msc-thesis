def report(results, name):
    lines = []

    def out(line=""):
        print(line)
        lines.append(line)

    def save():
        results.mkdir(parents=True, exist_ok=True)
        path = results / f"{name}.txt"
        path.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {path}")

    return out, save
