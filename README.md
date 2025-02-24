# Product Quantization

A TypeScript implementation of Product Quantization (PQ) for efficient similarity search in high-dimensional spaces.

## Installation

`npm install product-quantization`

## Overview

Product Quantization is a technique used to compress high-dimensional vectors into compact codes while preserving the ability to compute approximate distances. This makes it particularly useful for applications like similarity search in large-scale datasets.

For more detailed explanations of Product Quantization, see:

- [Understanding HNSW and Product Quantization](https://weaviate.io/blog/ann-algorithms-hnsw-pq#kmeans-encoding-results)
- [Product Quantization Explained](https://www.pinecone.io/learn/series/faiss/product-quantization/)

## Features

- Configurable number of subvectors and centroids
- TypeScript implementation with full type support
- Efficient encoding and decoding of vectors
- Support for custom training data

## Usage

### Basic Example

```typescript
import { ProductQuantizer } from "product-quantization";
// Initialize the quantizer
const pq = new ProductQuantizer({
  dimension: 128, // Original vector dimension
  numSubvectors: 8, // Number of subvectors
  numCentroids: 256 // Number of centroids per subvector (default: 256)
});

// Train the quantizer with your data
const trainingData = [
  new Float32Array([/ your 128-dimensional vector /])
  // ... more training vectors
];
pq.train(trainingData);

// Encode a vector
const vector = new Float32Array([/ your vector /]);
const encoded = pq.encode(vector);

// Decode the vector
const decoded = pq.decode(encoded);
```

### API Reference

#### Constructor

`new ProductQuantizer(params: {
  dimension: number;        // Original vector dimension
  numSubvectors: number;    // Number of subvectors
  numCentroids?: number;    // Number of centroids per subvector (default: 256)
})`

#### Methods

- `train(data: Float32Array[]): void`

  - Trains the quantizer using the provided training data
  - Each vector in the training data must match the specified dimension

- `encode(vector: Float32Array): Uint8Array`

  - Encodes a vector into a compact representation
  - Returns a Uint8Array containing the codes

- `decode(codes: Uint8Array): Float32Array`

  - Decodes the compact representation back to the original space
  - Returns a Float32Array containing the reconstructed vector

- `export(): object`

  - Exports the quantizer configuration and codebooks

- `exportCodebooks(): Float32Array[][]`
  - Exports just the codebooks

## Performance Considerations

- The training phase uses k-means clustering and may take time for large datasets
- Encoding and decoding operations are relatively fast
- Memory usage depends on the number of subvectors and centroids

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
