'use client'
import React, { useRef, useState, useCallback, useEffect } from 'react'
import { motion, useSpring, useTransform, SpringOptions } from 'framer-motion'
import { cn } from '@/lib/utils'

type SpotlightProps = {
  className?: string
  size?: number
  springOptions?: SpringOptions
}

export function Spotlight({
  className,
  size = 400,
  springOptions = { bounce: 0 },
}: SpotlightProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isHovered, setIsHovered] = useState(false)
  const [parentElement, setParentElement] = useState<HTMLElement | null>(null)

  const mouseX = useSpring(0, springOptions)
  const mouseY = useSpring(0, springOptions)

  useEffect(() => {
    if (containerRef.current) {
      const parent = containerRef.current.parentElement
      if (parent) {
        parent.style.position = 'relative'
        parent.style.overflow = 'hidden'
        setParentElement(parent)
      }
    }
  }, [])

  const handleMouseMove = useCallback(
    (event: MouseEvent) => {
      if (!parentElement) return
      const { left, top } = parentElement.getBoundingClientRect()
      mouseX.set(event.clientX - left)
      mouseY.set(event.clientY - top)
    },
    [parentElement, mouseX, mouseY]
  )

  useEffect(() => {
    if (!parentElement) return

    const onMouseEnter = () => setIsHovered(true)
    const onMouseLeave = () => setIsHovered(false)

    parentElement.addEventListener('mousemove', handleMouseMove)
    parentElement.addEventListener('mouseenter', onMouseEnter)
    parentElement.addEventListener('mouseleave', onMouseLeave)

    return () => {
      parentElement.removeEventListener('mousemove', handleMouseMove)
      parentElement.removeEventListener('mouseenter', onMouseEnter)
      parentElement.removeEventListener('mouseleave', onMouseLeave)
    }
  }, [parentElement, handleMouseMove])

  return (
    <motion.div
      ref={containerRef}
      className={cn(
        'pointer-events-none absolute rounded-full bg-[radial-gradient(circle_at_center,rgba(0,240,255,0.15),transparent_80%)] blur-2xl transition-opacity duration-300',
        isHovered ? 'opacity-100' : 'opacity-0',
        className
      )}
      style={{
        width: size,
        height: size,
        left: mouseX,
        top: mouseY,
        transform: 'translate(-50%, -50%)',
      }}
    />
  )
}
