import { BaseEmbeddings } from '@cherrystudio/embedjs-interfaces';

type OllamaRequestError = Error & {
    status?: number;
};

type OllamaEmbeddingsOptions = {
    model: string;
    baseUrl: string;
    dimensions?: number;
    keepAlive?: string | number;
    truncate?: boolean;
    requestOptions?: Record<string, unknown>;
    headers?: Record<string, string>;
};

type OllamaEmbeddingsResponse = {
    embeddings?: unknown;
    embedding?: unknown;
};

export class OllamaEmbeddings extends BaseEmbeddings {
    private readonly model: string;
    private readonly baseUrl: string;
    private readonly dimensions?: number;
    private resolvedDimensions?: number;
    private readonly keepAlive?: string | number;
    private readonly truncate?: boolean;
    private readonly requestOptions?: Record<string, unknown>;
    private readonly headers?: Record<string, string>;

    constructor(options: OllamaEmbeddingsOptions) {
        super();

        this.model = options.model;
        this.baseUrl = options.baseUrl;
        this.dimensions = options.dimensions;
        this.keepAlive = options.keepAlive;
        this.truncate = options.truncate;
        this.requestOptions = options.requestOptions;
        this.headers = options.headers;
    }

    override async getDimensions(): Promise<number> {
        if (this.dimensions !== undefined) {
            return this.dimensions;
        }

        if (this.resolvedDimensions !== undefined) {
            return this.resolvedDimensions;
        }

        const sampleEmbedding = await this.embedQuery('sample');
        if (!(sampleEmbedding?.length)) {
            throw new Error('Ollama embedding response did not include an embedding vector');
        }

        this.resolvedDimensions = sampleEmbedding.length;
        return this.resolvedDimensions;
    }

    override async embedDocuments(texts: string[]): Promise<number[][]> {
        return this.embed(texts);
    }

    override async embedQuery(text: string): Promise<number[]> {
        const embeddings = await this.embed([text]);
        const embedding = embeddings[0];

        if (!(embedding?.length)) {
            throw new Error('Ollama embedding response did not include an embedding vector');
        }

        return embedding;
    }

    private async embed(texts: string[]): Promise<number[][]> {
        try {
            const response = await this.post('/api/embed', {
                model: this.model,
                input: texts.length === 1 ? texts[0] : texts,
                keep_alive: this.keepAlive,
                truncate: this.truncate,
                dimensions: this.dimensions,
                options: this.requestOptions,
            });

            return this.normalizeEmbeddings(response);
        } catch (currentApiError) {
            if (!this.shouldFallbackToLegacyApi(currentApiError)) {
                throw currentApiError;
            }

            try {
                return await Promise.all(texts.map((text) => this.embedWithLegacyApi(text)));
            } catch (legacyApiError) {
                throw new Error('Ollama embeddings request failed using both /api/embed and legacy /api/embeddings', {
                    cause: legacyApiError instanceof Error ? legacyApiError : currentApiError,
                });
            }
        }
    }

    private async embedWithLegacyApi(text: string): Promise<number[]> {
        const response = await this.post('/api/embeddings', {
            model: this.model,
            prompt: text,
            keep_alive: this.keepAlive,
            options: this.requestOptions,
        });
        const embeddings = this.normalizeEmbeddings(response);
        const embedding = embeddings[0];

        if (!(embedding?.length)) {
            throw new Error('Ollama legacy embeddings response did not include an embedding vector');
        }

        return embedding;
    }

    private async post(path: string, body: Record<string, unknown>): Promise<OllamaEmbeddingsResponse> {
        const headers = new Headers(this.headers);
        if (!headers.has('Content-Type')) {
            headers.set('Content-Type', 'application/json');
        }

        const response = await fetch(new URL(path, this.baseUrl).toString(), {
            method: 'POST',
            headers,
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            const message = await response.text();
            const error = new Error(message || `Ollama request failed with status ${response.status}`) as OllamaRequestError;
            error.status = response.status;
            throw error;
        }

        return (await response.json()) as OllamaEmbeddingsResponse;
    }

    private normalizeEmbeddings(response: OllamaEmbeddingsResponse): number[][] {
        const currentEmbeddings = response.embeddings;
        if (this.isEmbeddingMatrix(currentEmbeddings) && currentEmbeddings.length > 0) {
            return currentEmbeddings;
        }

        const legacyEmbedding = response.embedding;
        if (this.isEmbeddingVector(legacyEmbedding) && legacyEmbedding.length > 0) {
            return [legacyEmbedding];
        }

        throw new Error('Ollama embedding response did not include embeddings');
    }

    private isEmbeddingMatrix(value: unknown): value is number[][] {
        return Array.isArray(value) && value.every((item) => this.isEmbeddingVector(item));
    }

    private isEmbeddingVector(value: unknown): value is number[] {
        return Array.isArray(value) && value.every((item) => typeof item === 'number');
    }

    private shouldFallbackToLegacyApi(error: unknown): boolean {
        if (this.dimensions !== undefined || this.truncate !== undefined) {
            return false;
        }

        if (!(error instanceof Error)) {
            return false;
        }

        const status = (error as OllamaRequestError).status;
        return status === 404 || status === 405;
    }
}
