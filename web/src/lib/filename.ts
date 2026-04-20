function stampFilename(name: string): string {
    const d = new Date();
    const pad = (n: number) => String(n).padStart(2, "0");
    const stamp = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
    return name.replace(/\.min\.wgsl$/, `-${stamp}.min.wgsl`);
}

export function downloadTextFile(content: string, filename: string): void {
    const blob = new Blob([content], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = stampFilename(filename);
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
