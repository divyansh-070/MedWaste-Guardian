const socket = new WebSocket("ws://localhost:8000/ws/speech");

socket.onopen = () => console.log("Connected to speech WebSocket");
socket.onmessage = (event) => {
    document.getElementById("response-text").innerText = event.data;
};
socket.onerror = (error) => console.error("WebSocket error:", error);

async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            socket.send(event.data);
        }
    };

    mediaRecorder.start(500);  // Send audio every 500ms
}

async function sendTextRequest() {
    const textInput = document.getElementById("text-input").value;
    const formData = new FormData();
    formData.append("query", textInput);

    const response = await fetch("http://localhost:8000/process-text", {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    document.getElementById("response-text").innerText = data.message;
}

async function sendImageRequest() {
    const imageInput = document.getElementById("image-upload").files[0];
    const formData = new FormData();
    formData.append("file", imageInput);

    const response = await fetch("http://localhost:8000/process-image", {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    document.getElementById("response-text").innerText = JSON.stringify(data.detections, null, 2);
}
