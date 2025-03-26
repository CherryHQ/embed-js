import 'dotenv/config';
import { RAGApplicationBuilder } from '@cherrystudio/embedjs';
import { OpenAiEmbeddings } from '@cherrystudio/embedjs-openai';
import { WebLoader } from '@cherrystudio/embedjs-loader-web';
import { PineconeDb } from '@cherrystudio/embedjs-pinecone';

const llmApplication = await new RAGApplicationBuilder()
    .setEmbeddingModel(new OpenAiEmbeddings())
    .setVectorDatabase(
        new PineconeDb({
            projectName: 'test',
            namespace: 'dev',
            indexSpec: {
                serverless: {
                    cloud: 'aws',
                    region: 'us-east-1',
                },
            },
        }),
    )
    .build();

await llmApplication.addLoader(new WebLoader({ urlOrContent: 'https://en.wikipedia.org/wiki/Tesla,_Inc.' }));
console.log(await llmApplication.query('Who founded Tesla?'));
