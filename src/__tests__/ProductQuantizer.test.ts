import { ProductQuantizer } from "../pq";

describe("ProductQuantizer", () => {
  it("should initialize correctly", () => {
    const pq = new ProductQuantizer({ dimension: 128, numSubvectors: 8 });
    expect(pq).toBeDefined();
  });

  test("simple quantization with 2 centroids", () => {
    // Create a simple PQ with 2 centroids and 2 subvectors
    const pq = new ProductQuantizer({
      dimension: 4,
      numSubvectors: 2,
      numCentroids: 2
    });

    // Create training data with clearly separated clusters
    const trainingData = [
      new Float32Array([0, 0, 10, 10]), // Cluster 1: [0,0] and [10,10]
      new Float32Array([0, 0, 10, 10]),
      new Float32Array([5, 5, 0, 0]), // Cluster 2: [5,5] and [0,0]
      new Float32Array([5, 5, 0, 0])
    ];

    // Train the quantizer
    pq.train(trainingData);

    const codebooks = pq.exportCodebooks();

    console.log(codebooks);
    expect(codebooks.length).toBe(2);
    expect(codebooks[0].length).toBe(2);
    expect(codebooks[0].some((arr) => arr[0] === 0 && arr[1] === 0)).toBe(true);
    expect(codebooks[0].some((arr) => arr[0] === 5 && arr[1] === 5)).toBe(true);
    expect(codebooks[1].length).toBe(2);
    expect(codebooks[1].some((arr) => arr[0] === 0 && arr[1] === 0)).toBe(true);
    expect(codebooks[1].some((arr) => arr[0] === 10 && arr[1] === 10)).toBe(
      true
    );

    // Test vector to encode/decode
    const testVector = new Float32Array([0, 0, 10, 10]);

    // Encode and decode
    const encoded = pq.encode(testVector);
    const decoded = pq.decode(encoded);

    console.log(encoded);
    console.log(decoded);

    expect(decoded).toEqual(testVector);
  });

  test("8-dimensional quantization with 4 subvectors", () => {
    const pq = new ProductQuantizer({
      dimension: 8,
      numSubvectors: 4,
      numCentroids: 2
    });

    // Create training data with clear patterns
    // Each subvector is 2D and has two distinct clusters
    const trainingData = [
      new Float32Array([0, 0, 5, 5, 0, 0, 5, 5]), // Pattern A
      new Float32Array([0, 0, 5, 5, 0, 0, 5, 5]),
      new Float32Array([3, 3, 0, 0, 3, 3, 0, 0]), // Pattern B
      new Float32Array([3, 3, 0, 0, 3, 3, 0, 0])
    ];

    pq.train(trainingData);

    const codebooks = pq.exportCodebooks();

    // Verify codebook structure
    expect(codebooks.length).toBe(4); // 4 subvectors
    expect(codebooks[0].length).toBe(2); // 2 centroids per subvector

    // Test encoding and decoding
    const testVector = new Float32Array([0, 0, 5, 5, 0, 0, 5, 5]);
    const encoded = pq.encode(testVector);
    const decoded = pq.decode(encoded);

    expect(decoded).toEqual(testVector);
  });
});
