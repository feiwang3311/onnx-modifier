var modifier = modifier || {};

modifier.Modifier = class {
    constructor(view) {
        this.view = view;
        this.model = null;
        this.graphs = null;
        this.name2ModelNode = new Map();
        this.name2ViewNode = new Map();
        this.name2NodeStates = new Map();
        this.namedEdges = new Map();

        this.addedOutputs = new Set();
        this.addedInputs = new Set();
        this.addedNode = new Map();
        this.addNodeKey = 0;
        this.changedAttributes = new Map();
        this.initializerEditInfo = new Map();
        this.renameMap = new Map();
        this.reBatchInfo = new Map();
        this.changedInputInfo = new Map();

        this.downloadWithShapeInf = false;
        this.downloadWithCleanUp = false;

    }

    loadModelGraph(model, graphs) {
        this.model = model;
        this.graphs = graphs;
        this.graph = this.graphs[0];
        // this.analyzeModelGraph();
        this.originInputs = new Set();
        for (var inp of this.graph.inputs) {
            var input_orig_name = inp.arguments[0].original_name;
            this.originInputs.add(input_orig_name)
        }
        this.name2NodeStatesOrig = new Map();
        //make a name2NodeStates copy for reset, to cope with fault caused by removing mistake added ops
        for (const name of this.name2NodeStates.keys())
        {
            this.name2NodeStatesOrig.set(name, 'Exist');
        }
        this.updateAddNodeDropDown();
    }

    // TODO: add filter feature like here: https://www.w3schools.com/howto/howto_js_dropdown.asp
    updateAddNodeDropDown() {
        // update dropdown supported node lost
        var addNodeDropdown = this.view._host.document.getElementById('add-node-dropdown');
        for (const node of this.model.supported_nodes) {
            // node: [domain, op]
            var option = new Option(node[1], node[0] + ':' + node[1]);
            // console.log(option)
            addNodeDropdown.appendChild(option);
        }
    }

    getShapeTypeInfo(name) {
        for (var value_info of this.graph._value_info) {
            if (value_info.name == name && value_info.type && value_info.type.tensor_type) {
                var tensor_type = value_info.type.tensor_type;
                let shape = [];
                if (tensor_type.shape && tensor_type.shape.dim) {
                    shape = tensor_type.shape.dim.map((dim) => dim.dim_param ? dim.dim_param : dim.dim_value ? dim.dim_value : null);
                }
                var tensor_type = this.graph._context.createTensorType(tensor_type.elem_type, shape);
                return [tensor_type.shape, tensor_type.dataType];
            }
            break;
        }
        return null;
    }

    randomString(length, chars) {
        var result = '';
        for (var i = length; i > 0; --i) result += chars[Math.floor(Math.random() * chars.length)];
        return result;
    }


    try_get_node_name(op_type)
    {
        var node_id = (this.addNodeKey++).toString();  // in case input (onnx) node has no name
        var modelNodeName = 'custom_added_' + op_type + node_id;

        if (this.addedNode.has(modelNodeName) || this.name2NodeStates.get(modelNodeName) ){
            modelNodeName = this.randomString(16, 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ');
        }
        return modelNodeName;
    }

    // ======= Record modified info =======> //
    addNode(op_domain, op_type) {
        //avoid to add a existed name node
        var modelNodeName = this.try_get_node_name(op_type);
        var properties = new Map();
        properties.set('domain', op_domain);
        properties.set('op_type', op_type);
        properties.set('name', modelNodeName);
        this.addedNode.set(modelNodeName, new view.LightNodeInfo(properties));

        this.applyAndUpdateView();
    }

    // Add a DeliminatorOp on an edge (between two nodes)
    addDeliminatorOnEdge(edgeId, attributes) {
        // Parse the edge ID: edge_<fromNode>_TO_<toNode>_TENSOR_<tensorName>
        const match = edgeId.match(/^edge_(.+)_TO_(.+)_TENSOR_(.*)$/);
        if (!match) {
            console.error('Invalid edge ID format:', edgeId);
            return;
        }

        const fromNodeName = decodeURIComponent(match[1]);
        const toNodeName = decodeURIComponent(match[2]);
        const tensorName = decodeURIComponent(match[3]);

        // Create the DeliminatorOp node
        var modelNodeName = this.try_get_node_name('DeliminatorOp');
        var properties = new Map();
        properties.set('domain', 'custom');
        properties.set('op_type', 'DeliminatorOp');
        properties.set('name', modelNodeName);

        // Set attributes
        var nodeAttributes = new Map();
        nodeAttributes.set('is_begin', [attributes.is_begin.toString(), 'int64']);
        nodeAttributes.set('func_name', [attributes.func_name, 'string']);
        nodeAttributes.set('scheduling_config', [attributes.scheduling_config, 'string']);

        // Create unique output tensor name for the DeliminatorOp
        var deliminatorOutputName = modelNodeName + '_output';

        // Set inputs - the DeliminatorOp takes the original tensor as input
        var inputs = new Map();
        inputs.set('X', [[tensorName, false]]);  // [name, is_optional]

        // Set outputs - the DeliminatorOp produces a new tensor
        var outputs = new Map();
        outputs.set('Y', [[deliminatorOutputName, false]]);

        // Create the node info
        var nodeInfo = new view.LightNodeInfo(properties, nodeAttributes, inputs, outputs);
        this.addedNode.set(modelNodeName, nodeInfo);

        // Now we need to update the destination node to use the DeliminatorOp's output
        // instead of the original tensor
        // Find which input of the destination node uses this tensor and rename it
        var destNode = this.name2ModelNode.get(toNodeName);

        if (destNode && destNode.inputs) {
            for (var input of destNode.inputs) {
                var found = false;
                for (var i = 0; i < input.arguments.length; i++) {
                    var arg = input.arguments[i];
                    if (arg.name === tensorName || arg.original_name === tensorName) {
                        // This is the input we need to rename
                        var orig_arg_name = arg.original_name || arg.name;
                        if (!this.renameMap.get(toNodeName)) {
                            this.renameMap.set(toNodeName, new Map());
                        }
                        this.renameMap.get(toNodeName).set(orig_arg_name, deliminatorOutputName);
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
        }

        this.applyAndUpdateView();
    }

    // Delete a DeliminatorOp and reconnect its input to its output consumers
    deleteDeliminatorOp(nodeName) {
        // Get the node info
        const nodeInfo = this.addedNode.get(nodeName);
        if (!nodeInfo) {
            console.error('DeliminatorOp not found:', nodeName);
            return false;
        }

        // Get the output tensor name (what this DeliminatorOp produces)
        const deliminatorOutputName = nodeName + '_output';

        // Find and remove the rename mapping that uses this output
        // The renameMap maps: destNodeName -> Map(originalTensorName -> newTensorName)
        // We need to find entries where newTensorName === deliminatorOutputName
        for (const [destNodeName, renameEntries] of this.renameMap) {
            for (const [origTensorName, newTensorName] of renameEntries) {
                if (newTensorName === deliminatorOutputName) {
                    renameEntries.delete(origTensorName);
                    // If the map is now empty, remove the dest node entry
                    if (renameEntries.size === 0) {
                        this.renameMap.delete(destNodeName);
                    }
                    break;
                }
            }
        }

        // Remove the node from addedNode
        this.addedNode.delete(nodeName);

        // Refresh the view
        this.applyAndUpdateView();
        return true;
    }

    // Get DeliminatorOp attributes for move feature
    getDeliminatorOpAttributes(nodeName) {
        const nodeInfo = this.addedNode.get(nodeName);
        if (!nodeInfo || !nodeInfo.attributes) {
            return null;
        }

        const attrs = {};
        for (const [name, value] of nodeInfo.attributes) {
            // value is [valueString, type]
            attrs[name] = value[0];
        }
        return attrs;
    }

    // Find all DeliminatorOps and scoped ops for a given func_name
    findScopedOps(funcName) {
        const result = {
            deliminatorOps: [],  // All DeliminatorOps with this func_name
            scopedOps: [],       // Ops between begin and end delimiters
            beginDelims: [],     // is_begin = 1 delimiters
            endDelims: []        // is_begin = 0 delimiters
        };

        // Find all DeliminatorOps with this func_name
        const beginDelims = result.beginDelims;  // is_begin = 1
        const endDelims = result.endDelims;      // is_begin = 0

        for (const [nodeName, nodeInfo] of this.addedNode) {
            if (nodeInfo.properties && nodeInfo.properties.get('op_type') === 'DeliminatorOp') {
                const attrs = this.getDeliminatorOpAttributes(nodeName);
                if (attrs && attrs.func_name === funcName) {
                    result.deliminatorOps.push(nodeName);
                    if (attrs.is_begin === '1') {
                        beginDelims.push(nodeName);
                    } else {
                        endDelims.push(nodeName);
                    }
                }
            }
        }

        // Build adjacency maps for the graph
        // We need to find nodes that are downstream of beginDelims and upstream of endDelims
        const nodeOutputs = new Map();  // nodeName -> [downstream node names]
        const nodeInputs = new Map();   // nodeName -> [upstream node names]

        // Process existing graph nodes
        for (const node of this.graph._nodes) {
            const nodeName = node.modelNodeName || node.name;
            if (!nodeOutputs.has(nodeName)) nodeOutputs.set(nodeName, []);
            if (!nodeInputs.has(nodeName)) nodeInputs.set(nodeName, []);

            // Get output tensor names
            const outputTensors = new Set();
            if (node.outputs) {
                for (const output of node.outputs) {
                    for (const arg of output.arguments) {
                        outputTensors.add(arg.name);
                    }
                }
            }

            // Find downstream nodes (nodes that consume this node's outputs)
            for (const otherNode of this.graph._nodes) {
                const otherName = otherNode.modelNodeName || otherNode.name;
                if (otherName === nodeName) continue;
                if (otherNode.inputs) {
                    for (const input of otherNode.inputs) {
                        for (const arg of input.arguments) {
                            if (outputTensors.has(arg.name) || outputTensors.has(arg.original_name)) {
                                if (!nodeOutputs.get(nodeName).includes(otherName)) {
                                    nodeOutputs.get(nodeName).push(otherName);
                                }
                                if (!nodeInputs.has(otherName)) nodeInputs.set(otherName, []);
                                if (!nodeInputs.get(otherName).includes(nodeName)) {
                                    nodeInputs.get(otherName).push(nodeName);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Also consider added nodes (DeliminatorOps)
        for (const [addedName, nodeInfo] of this.addedNode) {
            if (!nodeOutputs.has(addedName)) nodeOutputs.set(addedName, []);
            if (!nodeInputs.has(addedName)) nodeInputs.set(addedName, []);

            // Get input tensor name
            let inputTensor = null;
            if (nodeInfo.inputs) {
                for (const [inputName, args] of nodeInfo.inputs) {
                    if (args && args.length > 0) {
                        inputTensor = args[0][0];  // [[tensorName, isOptional]]
                    }
                }
            }

            // Get output tensor name
            let outputTensor = null;
            if (nodeInfo.outputs) {
                for (const [outputName, args] of nodeInfo.outputs) {
                    if (args && args.length > 0) {
                        outputTensor = args[0][0];
                    }
                }
            }

            // Find upstream node (who produces inputTensor)
            if (inputTensor) {
                for (const node of this.graph._nodes) {
                    const nodeName = node.modelNodeName || node.name;
                    if (node.outputs) {
                        for (const output of node.outputs) {
                            for (const arg of output.arguments) {
                                if (arg.name === inputTensor || arg.original_name === inputTensor) {
                                    if (!nodeInputs.get(addedName).includes(nodeName)) {
                                        nodeInputs.get(addedName).push(nodeName);
                                    }
                                    if (!nodeOutputs.has(nodeName)) nodeOutputs.set(nodeName, []);
                                    if (!nodeOutputs.get(nodeName).includes(addedName)) {
                                        nodeOutputs.get(nodeName).push(addedName);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Find downstream node (who consumes outputTensor via renameMap)
            if (outputTensor) {
                for (const [destNodeName, renameEntries] of this.renameMap) {
                    for (const [origName, newName] of renameEntries) {
                        if (newName === outputTensor) {
                            if (!nodeOutputs.get(addedName).includes(destNodeName)) {
                                nodeOutputs.get(addedName).push(destNodeName);
                            }
                            if (!nodeInputs.has(destNodeName)) nodeInputs.set(destNodeName, []);
                            if (!nodeInputs.get(destNodeName).includes(addedName)) {
                                nodeInputs.get(destNodeName).push(addedName);
                            }
                        }
                    }
                }
            }
        }

        // BFS from begin delimiters to find all reachable nodes
        const reachableFromBegin = new Set();
        const queue = [...beginDelims];
        while (queue.length > 0) {
            const current = queue.shift();
            if (reachableFromBegin.has(current)) continue;
            reachableFromBegin.add(current);
            const downstream = nodeOutputs.get(current) || [];
            for (const next of downstream) {
                if (!reachableFromBegin.has(next)) {
                    queue.push(next);
                }
            }
        }

        // BFS backwards from end delimiters to find all nodes that can reach them
        const canReachEnd = new Set();
        const queue2 = [...endDelims];
        while (queue2.length > 0) {
            const current = queue2.shift();
            if (canReachEnd.has(current)) continue;
            canReachEnd.add(current);
            const upstream = nodeInputs.get(current) || [];
            for (const prev of upstream) {
                if (!canReachEnd.has(prev)) {
                    queue2.push(prev);
                }
            }
        }

        // Scoped ops are those reachable from begin AND can reach end
        // Exclude the deliminator ops themselves
        for (const nodeName of reachableFromBegin) {
            if (canReachEnd.has(nodeName) && !result.deliminatorOps.includes(nodeName)) {
                result.scopedOps.push(nodeName);
            }
        }

        return result;
    }

    // Replace DeliminatorOps and scoped ops with a single Partition op
    replaceWithPartitionOp(funcName) {
        // Step 2: Use highlighting functionality to find all ops to replace
        const scopeInfo = this.findScopedOps(funcName);
        const allNodesToReplace = new Set([...scopeInfo.deliminatorOps, ...scopeInfo.scopedOps]);
        const beginDelims = scopeInfo.beginDelims;
        const endDelims = scopeInfo.endDelims;

        console.log('=== Replace with Partition ===');
        console.log('funcName:', funcName);
        console.log('Begin delims:', beginDelims);
        console.log('End delims:', endDelims);
        console.log('Nodes to replace:', [...allNodesToReplace]);

        if (allNodesToReplace.size === 0) {
            console.log('No nodes found for func_name:', funcName);
            return false;
        }

        // Step 3a: Get inputs of is_begin deliminator ops
        const partitionInputs = [];
        for (const delimName of beginDelims) {
            const nodeInfo = this.addedNode.get(delimName);
            if (nodeInfo && nodeInfo.inputs) {
                for (const [inputName, args] of nodeInfo.inputs) {
                    for (const arg of args) {
                        partitionInputs.push(arg[0]);  // tensor name
                    }
                }
            }
        }
        console.log('Partition inputs:', partitionInputs);

        // Step 3b: Get outputs of !is_begin (end) deliminator ops
        // These will be replaced by Partition's outputs
        const endDelimOutputs = [];  // [{delimName, tensorName, origTensorName}]
        for (const delimName of endDelims) {
            const nodeInfo = this.addedNode.get(delimName);
            if (nodeInfo && nodeInfo.outputs) {
                for (const [outputName, args] of nodeInfo.outputs) {
                    for (const arg of args) {
                        endDelimOutputs.push({
                            delimName: delimName,
                            tensorName: arg[0]  // the output tensor of the end delim
                        });
                    }
                }
            }
        }
        console.log('End delim outputs:', endDelimOutputs);

        // Step 1: Create the Partition op
        const partitionNodeName = this.try_get_node_name('Partition');
        const properties = new Map();
        properties.set('domain', 'custom');
        properties.set('op_type', 'Partition');
        properties.set('name', partitionNodeName);

        // Set func_name as an attribute
        const nodeAttributes = new Map();
        nodeAttributes.set('func_name', [funcName, 'string']);

        // Set inputs - use 'X' to match schema expectation
        // For now, use first input (if multiple begin delims, take first one)
        const inputs = new Map();
        if (partitionInputs.length > 0) {
            inputs.set('X', [[partitionInputs[0], false]]);
        }

        // Set outputs - use 'Y' to match schema expectation
        // For now, use first output (if multiple end delims, take first one)
        const outputs = new Map();
        const outputMapping = new Map();  // endDelimOutput -> partitionOutput
        const partitionOutputName = partitionNodeName + '_output';
        if (endDelimOutputs.length > 0) {
            outputs.set('Y', [[partitionOutputName, false]]);
            // Map all end delim outputs to this single partition output
            for (const endOut of endDelimOutputs) {
                outputMapping.set(endOut.tensorName, partitionOutputName);
            }
        }

        // Create the node info
        const nodeInfo = new view.LightNodeInfo(properties, nodeAttributes, inputs, outputs);
        this.addedNode.set(partitionNodeName, nodeInfo);

        console.log('Created Partition op:', partitionNodeName);
        console.log('  Inputs:', [...inputs.entries()].map(([k,v]) => k + '=' + JSON.stringify(v)));
        console.log('  Outputs:', [...outputs.entries()].map(([k,v]) => k + '=' + JSON.stringify(v)));
        console.log('  Output mapping (endDelim output -> partition output):', [...outputMapping.entries()]);

        // Step 3c: Update renameMap to redirect end delim outputs to Partition outputs
        // Find all nodes that consume end delim outputs and update their renameMap
        for (const [destNodeName, renameEntries] of this.renameMap) {
            if (allNodesToReplace.has(destNodeName)) continue;
            for (const [origName, newName] of renameEntries) {
                if (outputMapping.has(newName)) {
                    // This node was consuming an end delim output, redirect to Partition output
                    const partitionOutput = outputMapping.get(newName);
                    console.log('Redirecting renameMap[' + destNodeName + '][' + origName + ']: ' + newName + ' -> ' + partitionOutput);
                    renameEntries.set(origName, partitionOutput);
                }
            }
        }

        // Step 4: Remove all highlighted ops
        for (const nodeName of allNodesToReplace) {
            if (this.addedNode.has(nodeName)) {
                this.addedNode.delete(nodeName);
            } else {
                this.name2NodeStates.set(nodeName, 'Deleted');
            }
        }

        // Refresh the view
        this.applyAndUpdateView();
        return true;
    }

    // Pattern-based deliminator insertion
    applyPatterns(patterns) {
        let totalDelimitorsAdded = 0;
        let totalMatches = 0;

        for (const pattern of patterns) {
            let matches = [];
            if (pattern.type === 'sequence') {
                matches = this.findSequenceMatches(pattern);
            } else if (pattern.type === 'dag') {
                matches = this.findDagMatches(pattern);
            }

            // Add deliminators for each match
            matches.forEach((match, index) => {
                const funcName = matches.length > 1 ? `${pattern.name}_${index}` : pattern.name;
                const delimitorsAdded = this.addDelimitorsToMatch(match, pattern.scheduling_config, funcName);
                totalDelimitorsAdded += delimitorsAdded;
            });

            totalMatches += matches.length;
        }

        if (totalDelimitorsAdded > 0) {
            this.applyAndUpdateView();
        }

        return { total: totalDelimitorsAdded, matches: totalMatches };
    }

    // Find sequence pattern matches in the graph
    findSequenceMatches(pattern) {
        const matches = [];
        const ops = pattern.ops;
        const nodes = this.getNodeList();

        // Helper to get the op type string from a node
        const getNodeType = (node) => {
            if (typeof node.type === 'string') return node.type;
            if (node.type && node.type.name) return node.type.name;
            if (node.op_type) return node.op_type;
            return null;
        };

        // Build adjacency info
        const nodeOutputs = new Map();
        const tensorToConsumer = new Map();

        for (const node of nodes) {
            const nodeName = node.name;
            const outputs = [];
            if (node.outputs) {
                for (const output of node.outputs) {
                    for (const arg of output.arguments) {
                        outputs.push(arg.name);
                    }
                }
            }
            nodeOutputs.set(nodeName, outputs);

            if (node.inputs) {
                for (const input of node.inputs) {
                    for (const arg of input.arguments) {
                        if (!tensorToConsumer.has(arg.name)) {
                            tensorToConsumer.set(arg.name, []);
                        }
                        tensorToConsumer.get(arg.name).push(nodeName);
                    }
                }
            }
        }

        const getSuccessors = (nodeName) => {
            const successors = [];
            const outputs = nodeOutputs.get(nodeName) || [];
            for (const tensor of outputs) {
                const consumers = tensorToConsumer.get(tensor) || [];
                successors.push(...consumers);
            }
            return [...new Set(successors)];
        };

        const matchesOp = (node, opPattern) => {
            const nodeType = getNodeType(node);
            if (!nodeType) return false;
            const nodeTypeLower = nodeType.toLowerCase();

            let result = false;
            if (typeof opPattern === 'string') {
                result = nodeTypeLower === opPattern.toLowerCase();
            } else if (Array.isArray(opPattern)) {
                result = opPattern.some(op => nodeTypeLower === op.toLowerCase());
            } else if (opPattern.op) {
                const opList = Array.isArray(opPattern.op) ? opPattern.op : [opPattern.op];
                result = opList.some(op => nodeTypeLower === op.toLowerCase());
            }
            return result;
        };

        const isOptional = (opPattern) => {
            return typeof opPattern === 'object' && opPattern.optional === true;
        };

        const tryMatch = (startNode, opIndex, currentMatch) => {
            if (opIndex >= ops.length) return [currentMatch.slice()];

            const opPattern = ops[opIndex];

            if (matchesOp(startNode, opPattern)) {
                currentMatch.push(startNode);
                const successors = getSuccessors(startNode.name);

                if (opIndex === ops.length - 1) {
                    const result = [currentMatch.slice()];
                    currentMatch.pop();
                    return result;
                }

                const results = [];
                for (const succName of successors) {
                    const succNode = nodes.find(n => n.name === succName);
                    if (succNode) results.push(...tryMatch(succNode, opIndex + 1, currentMatch));
                }
                currentMatch.pop();
                return results;
            } else if (isOptional(opPattern)) {
                return tryMatch(startNode, opIndex + 1, currentMatch);
            }
            return [];
        };

        const usedNodes = new Set();
        for (const node of nodes) {
            if (usedNodes.has(node.name)) continue;
            const nodeMatches = tryMatch(node, 0, []);
            for (const match of nodeMatches) {
                const matchNodeNames = match.map(n => n.name);
                if (!matchNodeNames.some(name => usedNodes.has(name))) {
                    matches.push(match);
                    matchNodeNames.forEach(name => usedNodes.add(name));
                }
            }
        }
        return matches;
    }

    // Find DAG pattern matches
    findDagMatches(pattern) {
        const matches = [];
        const nodes = this.getNodeList();
        const patternNodes = pattern.nodes;
        const patternEdges = pattern.edges;

        const nodeOutputs = new Map();
        const tensorToConsumer = new Map();

        for (const node of nodes) {
            const outputs = [];
            if (node.outputs) {
                for (const output of node.outputs) {
                    for (const arg of output.arguments) {
                        outputs.push(arg.name);
                    }
                }
            }
            nodeOutputs.set(node.name, outputs);

            if (node.inputs) {
                for (const input of node.inputs) {
                    for (const arg of input.arguments) {
                        if (!tensorToConsumer.has(arg.name)) {
                            tensorToConsumer.set(arg.name, []);
                        }
                        tensorToConsumer.get(arg.name).push(node.name);
                    }
                }
            }
        }

        const matchesType = (graphNode, patternType) => {
            if (typeof patternType === 'string') return graphNode.type === patternType;
            if (Array.isArray(patternType)) return patternType.includes(graphNode.type);
            return false;
        };

        const hasEdge = (n1Name, n2Name) => {
            const outputs = nodeOutputs.get(n1Name) || [];
            for (const tensor of outputs) {
                const consumers = tensorToConsumer.get(tensor) || [];
                if (consumers.includes(n2Name)) return true;
            }
            return false;
        };

        const patternNodeIds = Object.keys(patternNodes);

        const tryAssignment = (assignment, patternIndex) => {
            if (patternIndex >= patternNodeIds.length) {
                for (const [from, to] of patternEdges) {
                    if (!hasEdge(assignment[from], assignment[to])) return [];
                }
                return [Object.assign({}, assignment)];
            }

            const patternNodeId = patternNodeIds[patternIndex];
            const patternType = patternNodes[patternNodeId];
            const results = [];

            for (const graphNode of nodes) {
                if (Object.values(assignment).includes(graphNode.name)) continue;
                if (matchesType(graphNode, patternType)) {
                    assignment[patternNodeId] = graphNode.name;
                    results.push(...tryAssignment(assignment, patternIndex + 1));
                    delete assignment[patternNodeId];
                }
            }
            return results;
        };

        const assignments = tryAssignment({}, 0);
        const usedNodes = new Set();

        for (const assignment of assignments) {
            const nodeNames = Object.values(assignment);
            if (!nodeNames.some(name => usedNodes.has(name))) {
                const matchNodes = nodeNames.map(name => nodes.find(n => n.name === name));
                matches.push({ nodes: matchNodes, assignment: assignment, pattern: pattern });
                nodeNames.forEach(name => usedNodes.add(name));
            }
        }
        return matches;
    }

    getNodeList() {
        const nodes = [];
        for (const [name, node] of this.name2ModelNode) {
            if (this.name2NodeStates.get(name) === 'Exist') nodes.push(node);
        }
        return nodes;
    }

    addDelimitorsToMatch(match, schedulingConfig, funcName) {
        let added = 0;
        let matchNodes, inputNodes, outputNodes;

        if (Array.isArray(match)) {
            matchNodes = match;
            inputNodes = [match[0]];
            outputNodes = [match[match.length - 1]];
        } else {
            matchNodes = match.nodes;
            const pattern = match.pattern;
            inputNodes = (pattern.inputs || []).map(id => {
                const nodeName = match.assignment[id];
                return matchNodes.find(n => n.name === nodeName);
            }).filter(n => n);
            outputNodes = (pattern.outputs || []).map(id => {
                const nodeName = match.assignment[id];
                return matchNodes.find(n => n.name === nodeName);
            }).filter(n => n);
        }

        const matchNodeNames = new Set(matchNodes.map(n => n.name));

        // Add is_begin=true deliminators on entry edges
        for (const node of inputNodes) {
            if (node.inputs) {
                for (const input of node.inputs) {
                    for (const arg of input.arguments) {
                        const tensorName = arg.name;
                        const producerNode = this.findTensorProducer(tensorName);
                        if (!producerNode || !matchNodeNames.has(producerNode)) {
                            this.addDeliminatorForPatternEdge(tensorName, node.name, funcName, schedulingConfig, true);
                            added++;
                        }
                    }
                }
            }
        }

        // Add is_begin=false deliminators on exit edges
        for (const node of outputNodes) {
            if (node.outputs) {
                for (const output of node.outputs) {
                    for (const arg of output.arguments) {
                        const tensorName = arg.name;
                        const consumers = this.findTensorConsumers(tensorName);
                        for (const consumer of consumers) {
                            if (!matchNodeNames.has(consumer)) {
                                this.addDeliminatorForPatternEdge(tensorName, consumer, funcName, schedulingConfig, false);
                                added++;
                            }
                        }
                    }
                }
            }
        }
        return added;
    }

    findTensorProducer(tensorName) {
        for (const [nodeName, node] of this.name2ModelNode) {
            if (this.name2NodeStates.get(nodeName) !== 'Exist') continue;
            if (node.outputs) {
                for (const output of node.outputs) {
                    for (const arg of output.arguments) {
                        if (arg.name === tensorName) return nodeName;
                    }
                }
            }
        }
        return null;
    }

    findTensorConsumers(tensorName) {
        const consumers = [];
        for (const [nodeName, node] of this.name2ModelNode) {
            if (this.name2NodeStates.get(nodeName) !== 'Exist') continue;
            if (node.inputs) {
                for (const input of node.inputs) {
                    for (const arg of input.arguments) {
                        if (arg.name === tensorName) consumers.push(nodeName);
                    }
                }
            }
        }
        return consumers;
    }

    addDeliminatorForPatternEdge(tensorName, toNodeName, funcName, schedulingConfig, isBegin) {
        var modelNodeName = this.try_get_node_name('DeliminatorOp');
        var properties = new Map();
        properties.set('domain', 'custom');
        properties.set('op_type', 'DeliminatorOp');
        properties.set('name', modelNodeName);

        var nodeAttributes = new Map();
        nodeAttributes.set('is_begin', [isBegin ? '1' : '0', 'int64']);
        nodeAttributes.set('func_name', [funcName, 'string']);
        nodeAttributes.set('scheduling_config', [schedulingConfig, 'string']);

        var deliminatorOutputName = modelNodeName + '_output';
        var inputs = new Map();
        inputs.set('X', [[tensorName, false]]);
        var outputs = new Map();
        outputs.set('Y', [[deliminatorOutputName, false]]);

        var nodeInfo = new view.LightNodeInfo(properties, nodeAttributes, inputs, outputs);
        this.addedNode.set(modelNodeName, nodeInfo);

        var destNode = this.name2ModelNode.get(toNodeName);
        if (destNode && destNode.inputs) {
            for (var input of destNode.inputs) {
                var found = false;
                for (var i = 0; i < input.arguments.length; i++) {
                    var arg = input.arguments[i];
                    if (arg.name === tensorName || arg.original_name === tensorName) {
                        var orig_arg_name = arg.original_name || arg.name;
                        if (!this.renameMap.get(toNodeName)) {
                            this.renameMap.set(toNodeName, new Map());
                        }
                        this.renameMap.get(toNodeName).set(orig_arg_name, deliminatorOutputName);
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
        }
    }

    addModelOutput(node_name) {
        var modelNode = this.name2ModelNode.get(node_name);
        // use a output argument as a proxy
        var output_name = modelNode.outputs[0].arguments[0].name;
        if (this.name2NodeStates.get("out_" + output_name)) {
            this.recoverSingleNode("out_" + output_name);
        } else {
            this.addedOutputs.add(output_name);
        }
        this.applyAndUpdateView();
    }

    addModelInput(input_name, input_shape_type) {
        this.name2NodeStates.set(input_name, 'Exist');
        this.addedInputs.add([input_name, input_shape_type]);
        this.applyAndUpdateView();
    }

    changeModelInput(input_name, input_shape_type) {
        // console.log(input_name, input_shape_type);
        this.deleteModelInput(input_name);
        this.addModelInput(input_name, input_shape_type);
    }

    deleteModelOutput(output_name) {
        this.name2NodeStates.set(output_name, 'Deleted');  // "out_" + xxx
        this.applyAndUpdateView();
    }

    deleteModelInput(input_name) {
        if (this.changedInputInfo.has(input_name)) {
            this.changedInputInfo.delete(input_name)
        }
        this.name2NodeStates.set(input_name, 'Deleted');
        this.applyAndUpdateView();
    }

    deleteSingleNode(node_name) {
        this.name2NodeStates.set(node_name, 'Deleted');
        this.name2ViewNode.get(node_name).element.style.opacity = 0.3;
        // this.deleteInputbyNode(node_name)
    }

    deleteNodeWithChildren(node_name) {
        if (this.name2NodeStates.get(node_name) == 'Deleted') return;

        this.name2NodeStates.set(node_name, 'Deleted');
        this.name2ViewNode.get(node_name).element.style.opacity = 0.3;
        // this.deleteInputbyNode(node_name)

        if (!this.namedEdges.has(node_name)) return; // for leaf node

        for (var i = 0; i < this.namedEdges.get(node_name).length; i++) {
            this.deleteNodeWithChildren(this.namedEdges.get(node_name)[i]);
        }
    }

    recoverSingleNode(node_name) {
        this.name2NodeStates.set(node_name, 'Exist');
        this.name2ViewNode.get(node_name).element.style.opacity = 1;
    }

    getOriginalName(param_type, modelNodeName, param_index, arg_index) {
        if (param_type == 'model_input') {
            var orig_arg_name = this.name2ModelNode.get(modelNodeName).arguments[0].original_name;
        }

        if (param_type == 'model_output') {
            // modelNodeName = 'out_' + modelNodeName
            // console.log(modelNodeName)
            var orig_arg_name = this.name2ModelNode.get(modelNodeName).arguments[0].original_name;
            // console.log(orig_arg_name)
        }

        if (param_type == 'input') {
            var orig_arg_name = this.name2ModelNode.get(modelNodeName).inputs[param_index].arguments[arg_index].original_name;
            // console.log(orig_arg_name)
        }
        if (param_type == 'output') {
            var orig_arg_name = this.name2ModelNode.get(modelNodeName).outputs[param_index].arguments[arg_index].original_name;
            // console.log(orig_arg_name)
        }

        return orig_arg_name;
    }

    getNodeUpdateInputType(modelNodeName, arg_index){
        var outType = undefined
        if (this.addedNode.has(modelNodeName)) { // for custom added node
            var parameterName = Array.from(this.addedNode.get(modelNodeName).inputs.keys())[arg_index]
            if (this.addedNode.get(modelNodeName).inputs.has(parameterName)) {
                var arg_name = this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0];  // [arg.name, arg.is_optional]
                // update the corresponding initializer name
                if (this.initializerEditInfo.has(arg_name)) {
                    outType = this.initializerEditInfo.get(arg_name)[0];
                }
            }
        }
        return outType;
    }

    changeNodeInputOutput(modelNodeName, parameterName, param_type, param_index, arg_index, targetValue) {
        if (this.addedNode.has(modelNodeName)) {  // for custom added node
            if (this.addedNode.get(modelNodeName).inputs.has(parameterName)) {
                var arg_name = this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0];  // [arg.name, arg.is_optional]
                // update the corresponding initializer name
                if (this.initializerEditInfo.has(arg_name)) {
                    var init_val = this.initializerEditInfo.get(arg_name);
                    this.initializerEditInfo.set(targetValue, init_val);
                    this.initializerEditInfo.delete(arg_name);
                }
                this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0] = targetValue;
            }
            // console.log(this.initializerEditInfo)

            if (this.addedNode.get(modelNodeName).outputs.has(parameterName)) {
                this.addedNode.get(modelNodeName).outputs.get(parameterName)[arg_index][0] = targetValue;
            }
        }

        else {    // for the nodes in the original model
            var orig_arg_name = this.getOriginalName(param_type, modelNodeName, param_index, arg_index);
            // console.log(orig_arg_name)

            if (!this.renameMap.get(modelNodeName)) {
                this.renameMap.set(modelNodeName, new Map());
            }
            this.renameMap.get(modelNodeName).set(orig_arg_name, targetValue);
            // console.log(this._renameMap)
        }
        // this.view._updateGraph()

        this.applyAndUpdateView();
    }

    changeInitializer(modelNodeName, parameterName, param_type, param_index, arg_index, type, targetValue) {
        var orig_arg_name = this.getOriginalName(param_type, modelNodeName, param_index, arg_index);
        this.initializerEditInfo.set(orig_arg_name, [type, targetValue]);
        // this.view._updateGraph()

        this.applyAndUpdateView();
    }

    changeAddedNodeInitializer(modelNodeName, parameterName, param_type, param_index, arg_index, type, targetValue) {
        var arg_name = this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0];
        this.initializerEditInfo.set(arg_name, [type, targetValue]);
        // this.view._updateGraph()

        this.applyAndUpdateView();
    }

    changeNodeAttribute(modelNodeName, attributeName, targetValue, type) {
        if (this.addedNode.has(modelNodeName)) {
            this.addedNode.get(modelNodeName).attributes.set(attributeName, [targetValue, type]);
        }
        // console.log(this._addedNode)

        else {    // for the nodes in the original model
            if (!this.changedAttributes.get(modelNodeName)) {
                this.changedAttributes.set(modelNodeName, new Map());
            }
            this.changedAttributes.get(modelNodeName).set(attributeName, [targetValue, type]);

        }

        // this.view._updateGraph()
        this.applyAndUpdateView();
    }

    changeBatchSize(type, value) {
        if (type === "fixed") {
            this.reBatchInfo.set("type", "fixed");
            this.reBatchInfo.set("value", value);
        }
        else {  // dynamic
            this.reBatchInfo.set("type", "dynamic");
            this.reBatchInfo.set("value", "dynamic");
        }
    }

    onOffShapeInf(turnedOn) {
        if (turnedOn)  this.downloadWithShapeInf = true;
        else this.downloadWithShapeInf = false;
    }

    onOffCleanUp(turnedOn) {
        if (turnedOn)  this.downloadWithCleanUp= true;
        else this.downloadWithCleanUp = false;
    }
    // <======= Record modified info ======= //

    // ======= Apply modified info and update view =======> //
    deleteEnter() {
        this.applyAndUpdateView();
    }

    refreshModelInputOutput() {
        if(!this.graph)return;
        // console.log(this.modifier.renameMap)
        // console.log(this.graph.outputs)
        // create and add new in/output to graph
        this.graph.reset_custom_modified_outputs();
        this.graph.reset_custom_modified_inputs();
        for (var output_name of this.addedOutputs) {
            this.graph.add_output(output_name);
        }
        for (var input_name_shape of this.addedInputs) {
            this.graph.add_input(input_name_shape);
        }
        for (var input of this.graph.inputs) {
            var input_orig_name = input.arguments[0].original_name;
            if (this.renameMap.get(input_orig_name)) {
                var new_name = this.renameMap.get(input_orig_name).get(input_orig_name);
                var arg_with_new_name = this.graph._context.argument(new_name, input_orig_name);
                arg_with_new_name.type = this.graph._context.argument(input_orig_name).type;
                input.arguments[0] = arg_with_new_name;

                // change all the name of node input linked with model input meanwhile
                for (var node of this.graph.nodes) {
                    for (var node_input of node.inputs) {
                        for (const [index, element] of node_input.arguments.entries()) {
                            if (element.original_name == input_orig_name) {
                                var arg_with_new_name = this.graph._context.argument(new_name, element.original_name);

                                node_input.arguments[index] = arg_with_new_name;

                                // save the changed name into _renameMap
                                // as this modified _renamedMap, so refreshModelInputOutput() shoulf be called before refreshNodeArguments()
                                if (!this.renameMap.get(node.modelNodeName)) {
                                    this.renameMap.set(node.modelNodeName, new Map());
                                }

                                var orig_arg_name = element.original_name;
                                this.renameMap.get(node.modelNodeName).set(orig_arg_name, new_name);
                            }
                        }
                    }
                }
            }
        }
        // console.log(this.graph.outputs)
        for (var output of this.graph.outputs) {
            var output_orig_name = output.arguments[0].original_name;
            if (this.renameMap.get('out_' + output_orig_name)) {
                // for model input and output, node.modelNodeName == element.original_name
                var new_name = this.renameMap.get('out_' + output_orig_name).get(output_orig_name);
                // console.log(new_name)
                var arg_with_new_name = this.graph._context.argument(new_name, output_orig_name);

                output.arguments[0] = arg_with_new_name;

                // change all the name of node output linked with the model output meanwhile
                for (var node of this.graph.nodes) {
                    for (var node_output of node.outputs) {
                        for (const [index, element] of node_output.arguments.entries()) {
                            if (element.original_name == output_orig_name) {
                                // console.log(element.original_name)
                                var arg_with_new_name = this.graph._context.argument(new_name, element.original_name);

                                node_output.arguments[index] = arg_with_new_name;

                                // save the changed name into _renameMap
                                // as this modified _renamedMap, so refreshModelInputOutput() shoulf be called before refreshNodeArguments()
                                if (!this.renameMap.get(node.modelNodeName)) {
                                    this.renameMap.set(node.modelNodeName, new Map());
                                }

                                var orig_arg_name = element.original_name;
                                this.renameMap.get(node.modelNodeName).set(orig_arg_name, new_name);
                            }
                        }
                    }
                }
            }
        }

        for (var output of this.graph.outputs) {
            var output_orig_name = output.arguments[0].original_name;
            if (this.name2NodeStates.get('out_' + output_orig_name) == "Deleted") {
                this.graph.delete_output(output_orig_name);
            }
        }
        for (var inp of this.graph.inputs) {
            var input_orig_name = inp.arguments[0].original_name;
            if (this.name2NodeStates.get(input_orig_name) == "Deleted") {
                this.graph.delete_input(input_orig_name);
            }
        }
    }

    // re-generate the added node according to addedNode according to the latest addedNode
    refreshAddedNode() {
        if(!this.graph)return;
        this.graph.reset_custom_added_node();
        // for (const node_info of this.addedNode.values()) {
        // for (const [modelNodeName, node_info] of this.lastViewGraph.addedNode) {
        for (const [modelNodeName, node_info] of this.addedNode) {
            // console.log(node_info)
            var node = this.graph.make_custom_added_node(node_info);
            // console.log(node)

            for (const input of node.inputs) {
                var arg_list_info = [];
                for (const arg of input._arguments) {
                    arg_list_info.push([arg.name, arg.is_optional]);
                }
                this.addedNode.get(modelNodeName).inputs.set(input.name, arg_list_info);
            }

            for (const output of node.outputs) {
                var arg_list_info = [];
                for (const arg of output._arguments) {
                    arg_list_info.push([arg.name, arg.is_optional]);
                }
                this.addedNode.get(modelNodeName).outputs.set(output.name, arg_list_info);
            }

        }
    }

    // re-fresh node arguments in case the node inputs/outputs are changed
    refreshNodeArguments() {
        if(!this.graph)return;
        for (var node of this.graph._nodes) {
            const nodeRenameMap = this.renameMap.get(node.modelNodeName);

            // check inputs
            for (var input of node.inputs) {
                for (const [index, element] of input.arguments.entries()) {
                    const origName = element.original_name;
                    if (origName) {
                        const newName = nodeRenameMap ? nodeRenameMap.get(origName) : null;
                        if (newName) {
                            // Apply rename mapping
                            var arg_with_new_name = this.graph._context.argument(newName, origName);
                            input.arguments[index] = arg_with_new_name;
                        } else if (element.name !== origName) {
                            // No mapping but name was changed - restore to original
                            var arg_with_orig_name = this.graph._context.argument(origName, origName);
                            input.arguments[index] = arg_with_orig_name;
                        }
                    }
                }
            }

            // check outputs
            for (var output of node.outputs) {
                for (const [index, element] of output.arguments.entries()) {
                    const origName = element.original_name;
                    if (origName) {
                        const newName = nodeRenameMap ? nodeRenameMap.get(origName) : null;
                        if (newName) {
                            // Apply rename mapping
                            var arg_with_new_name = this.graph._context.argument(newName, origName);
                            output.arguments[index] = arg_with_new_name;
                        } else if (element.name !== origName) {
                            // No mapping but name was changed - restore to original
                            var arg_with_orig_name = this.graph._context.argument(origName, origName);
                            output.arguments[index] = arg_with_orig_name;
                        }
                    }
                }
            }
        }
        this.namedEdges = new Map();
    }

    refreshNodeAttributes() {
        for (const node_name of this.changedAttributes.keys()) {
            var attr_change_map = this.changedAttributes.get(node_name);
            var node = this.name2ModelNode.get(node_name);

            for (var i = 0; i < node._attributes.length; ++i) {
                if (attr_change_map.get(node._attributes[i].name)) {
                    // [val, type]
                    node._attributes[i]._value = attr_change_map.get(node._attributes[i].name)[0];
                }
            }
        }
    }

    clearInfo() {
        this.namedEdges = new Map();
        this.changedAttributes = new Map();
        this.initializerEditInfo = new Map();
        this.renameMap = new Map();
        this.reBatchInfo = new Map();
        this.InputInfo = new Map();
        // clear custom added nodes
        this.addedNode = new Map();
        this.addedInputs = new Set();
        this.addedOutputs = new Set();
        if (this.graph)
        {
            this.graph.reset_custom_added_node();
            this.graph.reset_custom_modified_outputs();
            this.graph.reset_custom_modified_inputs();
        }
        // reset load location
        var container = this.view._getElementById('graph');
        container.scrollLeft = 0;
        container.scrollTop = 0;
        this.view._zoom = 1;
        this.addNodeKey = 0;
        this.applyAndUpdateView();
    }

    resetGraph() {
        // reset node states
        this.name2NodeStates = new Map();
        for (const name of this.name2NodeStatesOrig.keys())
        {
            this.name2NodeStates.set(name, 'Exist');
        }

        // console.log(this.modifier.renameMap)
        // reset node inputs/outputs
        for (const changed_node_name of this.renameMap.keys()) {
            var node = this.name2ModelNode.get(changed_node_name);
            // console.log(node)
            // console.log(typeof node)
            // console.log(node.constructor.name)
            if (node.arguments) {   // model input or model output. Because they are purely onnx.Parameter
                // node.arguments[0] = this.graph._context.argument(node.modelNodeName);
                node.arguments[0] = this.graph._context.argument(node.arguments[0].original_name);
            }

            else {                   // model nodes
                //reset inputs
                for (var input of node.inputs) {
                    for (var i = 0; i < input.arguments.length; ++i) {
                        // console.log(input.arguments[i].original_name)
                        if (this.renameMap.get(node.modelNodeName).get(input.arguments[i].original_name)) {
                            input.arguments[i] = this.graph._context.argument(input.arguments[i].original_name);
                        }
                    }
                }

                // reset outputs
                for (var output of node.outputs) {
                    for (var i = 0; i < output.arguments.length; ++i) {
                        if (this.renameMap.get(node.modelNodeName).get(output.arguments[i].original_name)) {
                            output.arguments[i] = this.graph._context.argument(output.arguments[i].original_name);
                        }
                    }
                }

            }
        }
        this.clearInfo();
    }

    clearGraph() {
        this.name2NodeStates = new Map();
        this.name2ModelNode = new Map();
        this.name2ViewNode = new Map();
        this.clearInfo();
    }

    applyAndUpdateView() {
        this.refreshAddedNode();
        this.refreshModelInputOutput();
        this.refreshNodeArguments();
        this.refreshNodeAttributes();

        // this.graphs has been modified (inplace)
        this.view._updateGraph(this.model, this.graphs);
    }
    // <======= Apply modified info and update view ======= //

}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Modifier = modifier.Modifier;
}

