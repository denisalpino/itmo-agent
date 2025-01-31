from fastapi.responses import JSONResponse
import time
import json

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import HttpUrl, ValidationError
import uvicorn
from utils.logger import setup_logger
from schemas.request import PredictionRequest, PredictionResponse
import asyncio

from agent import run_agent


class CustomHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        await asyncio.sleep(0.1)
        logger = request.app.state.logger

        try:
            body = await request.body()
            try:
                body_text = body.decode(encoding='utf-8')
            except UnicodeDecodeError:
                body_text = body.decode(encoding='cp1251')
            await logger.info(
                f"Incoming request: {request.method} {request.url}\n"
                f"Request body: {body_text}"
            )
        except Exception as e:
            await logger.error(f"Error reading request body: {str(e)}")

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            if isinstance(response, JSONResponse):
                response_body = json.dumps(response.body)
            else:
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk
                try:
                    response_body = response_body.decode(encoding='utf-8')
                except UnicodeDecodeError:
                    response_body = response_body.decode(encoding='cp1251')

            await logger.info(
                f"Request completed: {request.method} {request.url}\n"
                f"Status: {response.status_code}\n"
                f"Response body: {response_body}\n"
                f"Duration: {process_time:.3f}s"
            )

            if isinstance(response, JSONResponse):
                return response

            return Response(
                content=response_body.encode('utf-8'),
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )

        except Exception as e:
            process_time = time.time() - start_time
            await logger.error(f"Error processing response: {str(e)}")
            return JSONResponse(
                content={"detail": "Internal server error"},
                status_code=500
            )


async def predict(request: Request):
    logger = request.app.state.logger
    try:
        body = await request.body()
        try:
            data = json.loads(body.decode('utf-8'))
        except UnicodeDecodeError:
            data = json.loads(body.decode('cp1251'))
        request_data = PredictionRequest(**data)

        await logger.info(f"Processing prediction request with id: {request_data.id}")

        query = request_data.query
        agent_response = await run_agent(query)  # Используем await

        response_data = PredictionResponse(
            id=request_data.id,
            answer=int(agent_response["answer"]) if agent_response["answer"] is not None else None,
            reasoning=agent_response["reasoning"],
            sources=[HttpUrl(url) for url in agent_response["sources"]]
        )

        await logger.info(f"Successfully processed request {request_data.id}")

        response_dict = response_data.model_dump()
        response_dict["sources"] = [str(url) for url in response_dict["sources"]]

        return JSONResponse(
            content=response_dict,
            status_code=200,
            headers={"Access-Control-Allow-Origin": "*"}
        )

    except ValidationError as e:
        await logger.error(f"Validation error: {str(e)}")
        return JSONResponse(content={"detail": json.loads(e.json())}, status_code=400)
    except json.JSONDecodeError as e:
        await logger.error(f"JSON decode error: {str(e)}")
        return JSONResponse(content={"detail": "Invalid JSON format"}, status_code=400)
    except Exception as e:
        await logger.error(f"Internal error: {str(e)}")
        return JSONResponse(content={"detail": "Internal server error"}, status_code=500)


routes = [
    Route("/api/request", predict, methods=["POST"])
]


async def startup():
    return await setup_logger()

app = Starlette(
    debug=True,
    routes=routes,
    middleware=[Middleware(CustomHeaderMiddleware)]
)


@app.on_event("startup")
async def on_startup():
    app.state.logger = await setup_logger()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)