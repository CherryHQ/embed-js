import 'dotenv/config';
import path from 'node:path';
import { RAGApplicationBuilder } from '@cherrystudio/embedjs';
import { LibSqlDb, LibSqlStore } from '@cherrystudio/embedjs-libsql';
import { OpenAi, OpenAiEmbeddings } from '@cherrystudio/embedjs-openai';
import { ConfluenceLoader } from '@cherrystudio/embedjs-loader-confluence';

const databasePath = path.resolve('./examples/confluence/data.db');
const llmApplication = await new RAGApplicationBuilder()
    .setStore(new LibSqlStore({ path: databasePath }))
    .setVectorDatabase(new LibSqlDb({ path: databasePath }))
    .setModel(new OpenAi({ modelName: 'gpt-4o' }))
    .setEmbeddingModel(new OpenAiEmbeddings())
    .build();

await llmApplication.addLoader(new ConfluenceLoader({ spaceName: 'DEMO' }));
console.log(await llmApplication.query('Who founded Tesla?'));
