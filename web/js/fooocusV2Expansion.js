import { app } from "../../scripts/app.js";

let nodes = [];
let myNodes = [];

let debug = false;

function setProperty(object, key, value) {
  try {
    object[key] = value;
  } catch(error) {
    console.error("Fooocus V2 Expansion:", error);
  }
}

app.registerExtension({
  name: "Fooocus.V2.Expansion",  // Use the same name as your extension in Python
  async afterConfigureGraph() {
    for (let node of nodes) {
      if (node.type == "FooocusV2Expansion") {
        if (debug) console.log("My node:", node);

        myNodes.push(node);

        for (let widget of node.widgets) {
          if (widget.name == "positive_prompt") setProperty(widget, "tooltip", "The positive prompt to be expanded using Fooocus V2 Expansion.");
          if (widget.name == "prompt_seed") setProperty(widget, "tooltip", "Seed for prompt expansion. -1 for random seed.");
          if (widget.name == "seed_behavior") setProperty(widget, "tooltip", "Behavior for handling the seed. \r\nRandom: always new seed (between 1 and 0xffffffffffffffff), \r\nFixed: uses the input seed, \r\nForward Prompt Seed: forwards the seed used for the prompt expansion, \r\nRandom Forward Prompt Seed: always new seed (between 1 and 0xffffffffffffffff) as base then forwards the seed used for the prompt expansion.");
          if (widget.name == "expand") setProperty(widget, "tooltip", "Enable or disable prompt expansion. If disabled, the original prompt will be returned.");
          if (widget.name == "log") setProperty(widget, "tooltip", "Enable or disable logging of expansion process in the console.");
        }
      }
    }

    app.api.addEventListener("Fooocus.V2.Expansion.updateSeed", (event) => {
      const newSeed = event.detail.prompt_seed;
      const id = event.detail.id;

      if (debug) console.log(`New Seed ${newSeed} for ${id} node`);

      for (let myNode of myNodes) {
        if (myNode.id == id) {
          const seedInput = myNode.widgets.find(widget => widget.name === "prompt_seed"); 
          seedInput.value = newSeed;
        }
      }
    });
  },
  async nodeCreated(nodeCreated) {
    nodes.push(nodeCreated);
  }
});
