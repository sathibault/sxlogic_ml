# StreamLogic Neural Network Toolbox

This Python module provides tools to help implement neural networks using the StreamLogic platform.

StreamLogic actually supports 3 different neural-network implementations with varing levels of support and performance characteristics:
* Torte CNN engine - a convolutional neural-network inference engine.  This is a generic core that executes one layer of the network at a time and stores intermediate data on-chip using available large-memory blocks.
* Torte Dense NN engine - dense neural-network inference engine.  This is a generic core that only supports dense neural networks and is primarily used for audio.  Torte CNN also supports dense layers, but this engine has higher performance.
* Streaming NN library - a set of operators that can be chained together to construct a streaming neural network pipeline.  Each layer operates independently in parallel.  Input and activation data flows through the network as a stream and there is no need to store the whole result of each layer to memory.  There are of course line buffers in between layers that use up memory blocks, but it does not require any large-memory blocks or external memory.  However, all of the network parameters are stored using embedded-memory blocks and each layer requires at least one multiplier.  Consequently, the applicability of this approach is limited by those resources.

This tool currently only targets the streaming NN library.

## Streaming neural networks

