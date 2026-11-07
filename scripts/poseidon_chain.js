const fs = require("fs");
const readline = require("readline");
const { buildPoseidon } = require("circomlibjs");

async function main() {
  const poseidon = await buildPoseidon();
  const F = poseidon.F;

  function commit(request) {
    let acc = BigInt(request.domain);
    for (const field of request.fields) {
      acc = F.toObject(poseidon([acc, BigInt(field)]));
    }
    return acc.toString();
  }

  function evaluate(input) {
    if (Array.isArray(input.requests)) {
      return input.requests.map(commit);
    }
    return commit(input);
  }

  function serialize(output) {
    return Array.isArray(output) ? JSON.stringify(output) : output;
  }

  if (process.argv.includes("--server")) {
    const lines = readline.createInterface({ input: process.stdin });
    lines.on("line", (line) => {
      if (line.trim().length === 0) {
        return;
      }
      try {
        process.stdout.write(serialize(evaluate(JSON.parse(line))) + "\n");
      } catch (error) {
        console.error(error);
        process.exitCode = 1;
        lines.close();
      }
    });
    return;
  }

  const raw = fs.readFileSync(0, "utf8");
  process.stdout.write(serialize(evaluate(JSON.parse(raw))) + "\n");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
