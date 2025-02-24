export declare class ProductQuantizer {
    private numSubvectors;
    private numCentroids;
    private dimension;
    private codebooks;
    constructor(params: {
        dimension: number;
        numSubvectors: number;
        numCentroids?: number;
    });
    export(): {
        dimension: number;
        numSubvectors: number;
        numCentroids: number;
        codebooks: Float32Array<ArrayBufferLike>[][];
    };
    exportCodebooks(): Float32Array<ArrayBufferLike>[][];
    train(data: Float32Array[]): void;
    encode(vector: Float32Array): Uint8Array;
    decode(codes: Uint8Array): Float32Array;
}
