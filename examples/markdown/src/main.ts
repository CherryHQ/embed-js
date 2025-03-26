import 'dotenv/config';
import { LocalPathLoader, RAGApplicationBuilder } from '@cherrystudio/embedjs';
import { OpenAi, OpenAiEmbeddings } from '@cherrystudio/embedjs-openai';
import { HNSWDb } from '@cherrystudio/embedjs-hnswlib';

const llmApplication = await new RAGApplicationBuilder()
    .setModel(new OpenAi({ modelName: 'gpt-4o' }))
    .setEmbeddingModel(new OpenAiEmbeddings())
    .setVectorDatabase(new HNSWDb())
    .build();

await llmApplication.addLoader(new LocalPathLoader({ path: './docs' }));
console.log(await llmApplication.query('How do you create an embedJs application?'));
