import { app } from "../../scripts/app.js";

let nodes = [];
let myNodes = [];

app.registerExtension({
  name: "Fooocus.V2.Expansion",  // Use the same name as your extension in Python
  async afterConfigureGraph() {
    for (let node of nodes) {
      if (node.type == "FooocusV2Expansion") {
        myNodes.push(node);
        // console.log("My node:", node)
      }
    }

    app.api.addEventListener("Fooocus.V2.Expansion.updateSeed", (event) => {
      const newSeed = event.detail.prompt_seed;
      const id = event.detail.id;

      // console.log(`New Seed ${newSeed} for ${id} node`);

      for (let myNode of myNodes) {
        if (myNode.id == id) {
          const seedInput = myNode.widgets.find(w => w.name === "prompt_seed"); 
          seedInput.value = newSeed;
        }
      }
    });
  },
  async nodeCreated(nodeCreated) {
    nodes.push(nodeCreated);
  }
});
