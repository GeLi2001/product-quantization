import { kmeans } from "ml-kmeans";

export class ProductQuantizer {
  private numSubvectors: number;
  private numCentroids: number;
  private dimension: number;
  private codebooks: Float32Array[][];

  constructor(params: {
    dimension: number;
    numSubvectors: number;
    numCentroids?: number; // Default number of centroids for 8-bit code
  }) {
    const { dimension, numSubvectors, numCentroids } = params;
    this.numSubvectors = numSubvectors;
    this.dimension = dimension;
    this.numCentroids = numCentroids ?? 256;
    this.codebooks = [];
  }

  public export() {
    return {
      dimension: this.dimension,
      numSubvectors: this.numSubvectors,
      numCentroids: this.numCentroids,
      codebooks: this.codebooks
    };
  }

  public exportCodebooks() {
    return this.codebooks;
  }

  public train(data: Float32Array[]): void {
    // check dimension match
    if (data.every((vector) => vector.length !== this.dimension)) {
      throw new Error(
        `All vectors must have the same dimension with ${this.dimension}`
      );
    }

    // Initialize codebooks array
    this.codebooks = new Array(this.numSubvectors);

    // Calculate dimensions per subvector
    const subvectorDim = Math.floor(this.dimension / this.numSubvectors);

    // For each subvector
    for (let i = 0; i < this.numSubvectors; i++) {
      // Extract subvectors from training data
      const subvectors = data.map((vector) => {
        const start = i * subvectorDim;
        let end = start + subvectorDim;
        if (i === this.numSubvectors - 1) {
          end = Math.min(end, this.dimension);
        }
        return Array.from(vector.slice(start, end));
      });

      // Train k-means on subvectors
      const result = kmeans(subvectors, this.numCentroids, {
        maxIterations: 100,
        tolerance: 1e-6
      });

      // Store centroids as Float32Array in codebook
      this.codebooks[i] = result.centroids.map((centroid) => {
        return new Float32Array(centroid);
      });
    }
  }

  public encode(vector: Float32Array): Uint8Array {
    const codes = new Uint8Array(this.numSubvectors);
    const subvectorDim = Math.floor(this.dimension / this.numSubvectors);

    // For each subvector
    for (let i = 0; i < this.numSubvectors; i++) {
      const start = i * subvectorDim;
      let end = start + subvectorDim;
      if (i === this.numSubvectors - 1) {
        end = Math.min(end, this.dimension);
      }
      const subvector = vector.slice(start, end);

      // Find closest centroid
      let minDist = Infinity;
      let bestCode = 0;

      // Compare with each centroid in the codebook
      for (let j = 0; j < this.numCentroids; j++) {
        // Calculate Euclidean distance
        let dist = 0;
        for (let k = 0; k < subvector.length; k++) {
          const diff = subvector[k] - this.codebooks[i][j][k];
          dist += diff * diff;
        }

        if (dist < minDist) {
          minDist = dist;
          bestCode = j;
        }
      }

      codes[i] = bestCode;
    }

    return codes;
  }

  public decode(codes: Uint8Array): Float32Array {
    const decoded = [];
    for (let i = 0; i < this.numSubvectors; i++) {
      const centroid = this.codebooks[i][codes[i]];
      decoded.push(...centroid);
    }

    return new Float32Array(decoded);
  }
}
