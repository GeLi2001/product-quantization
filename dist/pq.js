"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ProductQuantizer = void 0;
const ml_kmeans_1 = require("ml-kmeans");
class ProductQuantizer {
    constructor(params) {
        const { dimension, numSubvectors, numCentroids } = params;
        this.numSubvectors = numSubvectors;
        this.dimension = dimension;
        this.numCentroids = numCentroids ?? 256;
        this.codebooks = [];
    }
    export() {
        return {
            dimension: this.dimension,
            numSubvectors: this.numSubvectors,
            numCentroids: this.numCentroids,
            codebooks: this.codebooks
        };
    }
    exportCodebooks() {
        return this.codebooks;
    }
    train(data) {
        // check dimension match
        if (data.every((vector) => vector.length !== this.dimension)) {
            throw new Error(`All vectors must have the same dimension with ${this.dimension}`);
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
            const result = (0, ml_kmeans_1.kmeans)(subvectors, this.numCentroids, {
                maxIterations: 100,
                tolerance: 1e-6
            });
            // Store centroids as Float32Array in codebook
            this.codebooks[i] = result.centroids.map((centroid) => {
                return new Float32Array(centroid);
            });
        }
    }
    encode(vector) {
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
    decode(codes) {
        const decoded = [];
        for (let i = 0; i < this.numSubvectors; i++) {
            const centroid = this.codebooks[i][codes[i]];
            decoded.push(...centroid);
        }
        return new Float32Array(decoded);
    }
}
exports.ProductQuantizer = ProductQuantizer;
