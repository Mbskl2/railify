import axios from "axios";

export const process = async (file: File) => {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post(
      "http://127.0.0.1:5000/api/process",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    const { png_data, svg_data } = response.data;

    downloadFile(png_data, "image/png", "output.png");
    downloadFile(svg_data, "image/svg+xml", "output.svg");
  } catch (error) {
    console.error("Error posting data:", error);
  }
};

function downloadFile(content: string, type: string, name: string) {
  const blobPart = base64ToArrayBuffer(content);
  const blob = new Blob([blobPart], { type });
  const link = document.createElement("a");
  link.href = window.URL.createObjectURL(blob);
  link.download = name;
  link.click();
}

export const base64ToArrayBuffer = (base64: any) => {
  const binaryString = window.atob(base64);
  const binaryLen = binaryString.length;
  const bytes = new Uint8Array(binaryLen);
  for (let i = 0; i < binaryLen; i++) {
    const ascii = binaryString.charCodeAt(i);
    bytes[i] = ascii;
  }
  return bytes;
};

export interface ReturnResult {
  fileContent: string;
  fileName: string;
  fileType: string;
}
