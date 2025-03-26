import 'dotenv/config';
import { RAGApplicationBuilder, UrlLoader } from '@cherrystudio/embedjs';
import { OpenAi, OpenAiEmbeddings } from '@cherrystudio/embedjs-openai';
import { HNSWDb } from '@cherrystudio/embedjs-hnswlib';

const llmApplication = await new RAGApplicationBuilder()
    .setModel(new OpenAi({ modelName: 'gpt-4o' }))
    .setEmbeddingModel(new OpenAiEmbeddings())
    .setVectorDatabase(new HNSWDb())
    .build();

await llmApplication.addLoader(new UrlLoader({ url: 'https://en.wikipedia.org/wiki/Tesla,_Inc.' }));

console.log(await llmApplication.query('Who founded Tesla?'));
